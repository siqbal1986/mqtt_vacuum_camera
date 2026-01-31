"""
Obstacle View Manager
Version: 2025.11.0
This module handles obstacle detection, image download, and rendering
for vacuums with ObstacleImagesCapability (e.g., Dreame vacuums).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from hashlib import sha1
import math
from pathlib import Path
from typing import Any, Callable, Optional

from homeassistant.core import Event, HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.debounce import Debouncer
from valetudo_map_parser.config.shared import CameraShared
from valetudo_map_parser.config.utils import ResizeParams, async_resize_image

from custom_components.mqtt_vacuum_camera.const import (
    DOWNLOAD_TIMEOUT,
    LOGGER,
    OBSTACLE_SEARCH_RADIUS_MULTIPLIER,
    CameraModes,
)


@dataclass
class ObstacleViewContext:
    """Context object containing dependencies for ObstacleView."""

    hass: HomeAssistant
    shared: CameraShared
    file_name: str
    storage_path: str
    download_image_func: Callable
    open_image_func: Callable
    pil_to_bytes_func: Callable


class ObstacleView:
    """
    Manages obstacle view functionality for vacuum cameras.

    This class handles:
    - Event listening for obstacle coordinate clicks
    - Finding nearest obstacles to clicked coordinates
    - Downloading obstacle images from the vacuum
    - Processing and resizing obstacle images
    - Managing camera mode transitions between MAP_VIEW and OBSTACLE_VIEW
    """

    def __init__(self, context: ObstacleViewContext):
        """
        Initialize the ObstacleView manager.

        Args:
            context: ObstacleViewContext containing all dependencies
        """
        self.hass = context.hass
        self._shared = context.shared
        self._file_name = context.file_name
        self._storage_path = Path(context.storage_path)
        self._entity_id: Optional[str] = None
        self._download_image = context.download_image_func
        self._open_image = context.open_image_func
        self._pil_to_bytes = context.pil_to_bytes_func
        self._obstacle_cache_dir = self._storage_path / "obstacles"

        # State management
        self._obstacle_image: Optional[bytes] = None
        self._processing: bool = False
        self._latest_obstacle_event: Optional[Event] = None
        self._prefetch_task: Optional[asyncio.Task] = None
        self._prefetch_signature: Optional[tuple[str, ...]] = None

        # Event listener handles
        self._event_listener = None

        # Initialize debouncer for obstacle view events
        self._debouncer = Debouncer(
            self.hass,
            LOGGER,
            cooldown=0.5,  # 500ms debounce for rapid events
            immediate=True,  # Process first event immediately
            function=self._process_obstacle_event,
        )

    async def async_setup(self, entity_id: str) -> None:
        """
        Set up the obstacle view manager and register event listeners.

        Args:
            entity_id: Camera entity ID for event filtering
        """
        self._entity_id = entity_id
        await asyncio.to_thread(
            self._obstacle_cache_dir.mkdir, parents=True, exist_ok=True
        )
        self._event_listener = self.hass.bus.async_listen(
            "mqtt_vacuum_camera_obstacle_coordinates",
            self._debounced_obstacle_handler,
        )
        LOGGER.debug("%s: ObstacleView manager initialized", self._file_name)

    async def async_cleanup(self) -> None:
        """Clean up event listeners and resources."""
        if self._event_listener:
            self._event_listener()
            self._event_listener = None

        if self._debouncer:
            self._debouncer.async_shutdown()

        if self._prefetch_task and not self._prefetch_task.done():
            self._prefetch_task.cancel()
            self._prefetch_task = None

        # Clear state
        self._obstacle_image = None
        self._processing = False

        LOGGER.debug("%s: ObstacleView manager cleaned up", self._file_name)

    @staticmethod
    def _obstacle_cache_key(obstacle: dict[str, Any]) -> Optional[str]:
        """Build a cache key for an obstacle entry."""
        link = obstacle.get("link")
        label = obstacle.get("label")
        if link:
            return link
        if label:
            return str(label)
        return None

    def _get_cache_path(self, obstacle: dict[str, Any]) -> Optional[Path]:
        """Return the cache path for an obstacle image."""
        cache_key = self._obstacle_cache_key(obstacle)
        if not cache_key:
            return None
        digest = sha1(cache_key.encode("utf-8")).hexdigest()
        return self._obstacle_cache_dir / f"{digest}.jpg"

    async def _load_cached_image(self, obstacle: dict[str, Any]) -> Optional[bytes]:
        """Load a cached obstacle image if it exists."""
        cache_path = self._get_cache_path(obstacle)
        if not cache_path or not cache_path.exists():
            return None
        try:
            return await asyncio.to_thread(cache_path.read_bytes)
        except OSError as err:
            LOGGER.warning(
                "%s: Failed to read cached obstacle image: %s",
                self._file_name,
                err,
                exc_info=True,
            )
            return None

    async def _save_cached_image(
        self, obstacle: dict[str, Any], image_data: bytes
    ) -> None:
        """Persist obstacle image bytes to cache."""
        cache_path = self._get_cache_path(obstacle)
        if not cache_path:
            return
        try:
            await asyncio.to_thread(cache_path.write_bytes, image_data)
        except OSError as err:
            LOGGER.warning(
                "%s: Failed to cache obstacle image: %s",
                self._file_name,
                err,
                exc_info=True,
            )

    async def async_prefetch_obstacles(
        self, obstacles: list[dict[str, Any]]
    ) -> None:
        """Prefetch obstacle images into the cache."""
        if not obstacles:
            return

        signature_items = []
        for obstacle in obstacles:
            cache_key = self._obstacle_cache_key(obstacle)
            if cache_key:
                signature_items.append(cache_key)
        signature = tuple(sorted(signature_items))
        if not signature or signature == self._prefetch_signature:
            return

        if self._prefetch_task and not self._prefetch_task.done():
            return

        self._prefetch_signature = signature
        self._prefetch_task = self.hass.async_create_task(
            self._async_prefetch_obstacles(obstacles)
        )

    async def _async_prefetch_obstacles(
        self, obstacles: list[dict[str, Any]]
    ) -> None:
        """Download missing obstacle images to the cache."""
        semaphore = asyncio.Semaphore(3)

        async def _download_if_missing(obstacle: dict[str, Any]) -> None:
            cache_path = self._get_cache_path(obstacle)
            if not cache_path or cache_path.exists() or not obstacle.get("link"):
                return
            async with semaphore:
                image_data = await self._download_image(
                    obstacle["link"], DOWNLOAD_TIMEOUT
                )
                if image_data:
                    await self._save_cached_image(obstacle, image_data)

        await asyncio.gather(*(_download_if_missing(o) for o in obstacles))

    async def _debounced_obstacle_handler(self, event: Event) -> None:
        """Handler that debounces incoming obstacle view events."""
        # Store the latest event data
        self._latest_obstacle_event = event
        # Trigger the debouncer - it will call _process_obstacle_event after cooldown
        await self._debouncer.async_call()

    async def _process_obstacle_event(self) -> None:
        """Process the latest obstacle event after debouncing."""
        if (
            not hasattr(self, "_latest_obstacle_event")
            or self._latest_obstacle_event is None
        ):
            return

        event = self._latest_obstacle_event
        await self.handle_obstacle_view(event)

    @staticmethod
    def find_nearest_obstacle(
        x: int,
        y: int,
        obstacles: list[dict[str, Any]],
        width: int,
        height: int,
    ) -> Optional[dict[str, Any]]:
        """
        Find the nearest obstacle to the given coordinates.

        Args:
            x: X coordinate of the click
            y: Y coordinate of the click
            obstacles: List of obstacle data dictionaries
            width: Image width for distance calculation
            height: Image height for distance calculation

        Returns:
            The nearest obstacle dictionary or None if no obstacle found within range
        """
        if height <= 0 or width <= 0:
            LOGGER.warning(
                "Invalid image dimensions: width=%d, height=%d", width, height
            )
            return None

        nearest_obstacle = None
        min_distance = max(
            1, round(OBSTACLE_SEARCH_RADIUS_MULTIPLIER * (width / height))
        )

        LOGGER.debug(
            "Finding in the nearest %d pixels obstacle to coordinates: %d, %d",
            min_distance,
            x,
            y,
        )

        for obstacle in obstacles:
            obstacle_point = obstacle["point"]
            obstacle_x = obstacle_point["x"]
            obstacle_y = obstacle_point["y"]

            distance = math.hypot(x - obstacle_x, y - obstacle_y)

            if distance < min_distance:
                min_distance = distance
                nearest_obstacle = obstacle

        return nearest_obstacle

    def _validate_obstacle_request(self, event: Event) -> bool:
        """
        Validate if obstacle request can be processed.

        Args:
            event: The event containing obstacle coordinates

        Returns:
            True if request is valid, False otherwise
        """
        # Prevent processing if already in obstacle processing modes
        if self._shared.camera_mode in [
            CameraModes.OBSTACLE_DOWNLOAD,
            CameraModes.OBSTACLE_SEARCH,
        ]:
            LOGGER.debug(
                "%s: Already processing obstacle view (mode: %s), ignoring request",
                self._file_name,
                self._shared.camera_mode,
            )
            return False

        # Validate we have obstacle data and are in MAP_VIEW
        if not self._shared.obstacles_data:
            LOGGER.debug("%s: No obstacles data available", self._file_name)
            return False

        if self._shared.camera_mode != CameraModes.MAP_VIEW:
            LOGGER.debug("%s: Not in MAP_VIEW mode", self._file_name)
            return False

        # Validate entity ID matches
        if event.data.get("entity_id") != self._entity_id:
            LOGGER.debug("%s: Entity ID mismatch", self._file_name)
            return False

        return True

    async def _set_camera_mode(self, mode: CameraModes, reason: str = "") -> None:
        """
        Set the camera mode and manage state transitions.

        Args:
            mode: The camera mode to set
            reason: Optional reason for the mode change (for logging)
        """
        self._shared.camera_mode = mode

        if mode == CameraModes.OBSTACLE_VIEW:
            self._shared.image_grab = False
            self._processing = True
        elif mode == CameraModes.MAP_VIEW:
            self._obstacle_image = None
            self._processing = False
            self._shared.image_grab = True

        log_msg = "%s: Camera Mode Change to %s"
        if reason:
            log_msg += ", %s"
            LOGGER.debug(log_msg, self._file_name, self._shared.camera_mode, reason)
        else:
            LOGGER.debug(log_msg, self._file_name, self._shared.camera_mode)

    async def handle_obstacle_view(self, event: Event) -> Optional[bytes]:
        """
        Handle the obstacle view event.

        This is the main entry point for obstacle view processing.

        Args:
            event: The event containing obstacle coordinates

        Returns:
            The obstacle image bytes or None
        """
        LOGGER.debug(
            "%s: Executing obstacle view logic for event: %s, Data: %s",
            self._file_name,
            str(event.event_type),
            str(event.data),
        )

        LOGGER.debug(
            "%s: Current camera mode: %s", self._file_name, self._shared.camera_mode
        )

        # Check if we are in obstacle view and switch back to map view
        if self._shared.camera_mode == CameraModes.OBSTACLE_VIEW:
            await self._set_camera_mode(
                CameraModes.MAP_VIEW, "Obstacle View Exit Requested"
            )
            return None

        # Validate state before processing
        if not self._validate_obstacle_request(event):
            return None

        # Extract coordinates from event
        coordinates = event.data.get("coordinates")
        if not coordinates:
            LOGGER.debug("%s: No coordinates provided", self._file_name)
            return None

        # Search and download obstacle image
        return await self._search_and_download_obstacle(
            coordinates.get("x"),
            coordinates.get("y"),
        )

    async def _search_and_download_obstacle(
        self, coord_x: int, coord_y: int
    ) -> Optional[bytes]:
        """
        Search for nearest obstacle and download its image.

        Args:
            coord_x: X coordinate of the click
            coord_y: Y coordinate of the click

        Returns:
            The obstacle image bytes or None
        """
        # Set mode to OBSTACLE_SEARCH
        await self._set_camera_mode(
            CameraModes.OBSTACLE_SEARCH, "Obstacle View Requested"
        )

        # Find the nearest obstacle
        nearest_obstacle = self.find_nearest_obstacle(
            coord_x,
            coord_y,
            self._shared.obstacles_data,
            self._shared.image_ref_width,
            self._shared.image_ref_height,
        )

        if not nearest_obstacle:
            LOGGER.debug("%s: No nearby obstacle found", self._file_name)
            await self._set_camera_mode(
                CameraModes.MAP_VIEW, "No nearby obstacle found"
            )
            return None

        LOGGER.debug(
            "%s: Nearest obstacle found: %r", self._file_name, nearest_obstacle
        )

        # Check if obstacle has a link
        if not nearest_obstacle.get("link"):
            LOGGER.debug("%s: No link found for the obstacle image", self._file_name)
            await self._set_camera_mode(
                CameraModes.MAP_VIEW, "No link found for the obstacle image"
            )
            return None

        # Download and process the obstacle image
        return await self._download_and_process(nearest_obstacle)

    async def _download_and_process(self, obstacle: dict[str, Any]) -> Optional[bytes]:
        """
        Download and process obstacle image.

        Args:
            obstacle: Obstacle data dictionary

        Returns:
            The processed obstacle image bytes or None
        """
        cached_image = await self._load_cached_image(obstacle)
        if cached_image is not None:
            LOGGER.debug("%s: Using cached obstacle image", self._file_name)
            return await self._process_obstacle_image(cached_image, obstacle)

        # Download the obstacle image
        await self._set_camera_mode(
            CameraModes.OBSTACLE_DOWNLOAD,
            f"Downloading image: {obstacle['link']}",
        )

        try:
            image_data = await asyncio.wait_for(
                fut=self._download_image(obstacle["link"], DOWNLOAD_TIMEOUT),
                timeout=(DOWNLOAD_TIMEOUT + 1),  # dead man switch
            )
        except asyncio.TimeoutError:
            LOGGER.warning("%s: Image download timed out", self._file_name)
            await self._set_camera_mode(
                CameraModes.MAP_VIEW, "Obstacle image download timed out"
            )
            return None

        if image_data is None:
            LOGGER.debug("%s: No image downloaded", self._file_name)
            await self._set_camera_mode(CameraModes.MAP_VIEW, "No image downloaded")
            return None

        await self._save_cached_image(obstacle, image_data)

        # Process the downloaded image
        return await self._process_obstacle_image(image_data, obstacle)

    async def _process_obstacle_image(
        self,
        image_data: bytes,
        obstacle: dict[str, Any],
    ) -> Optional[bytes]:
        """
        Process and resize the downloaded obstacle image.

        Args:
            image_data: Raw image data bytes
            obstacle: Obstacle data dictionary

        Returns:
            Processed image bytes or None on error
        """
        await self._set_camera_mode(
            CameraModes.OBSTACLE_VIEW, "Image downloaded successfully"
        )

        try:
            # Open the downloaded image with PIL
            pil_img = await self._open_image(image_data)

            # Resize the image if aspect ratio is provided
            aspect_ratio = self._shared.image_aspect_ratio
            if aspect_ratio != "None":
                resize_data = ResizeParams(
                    pil_img=pil_img,
                    width=self._shared.image_ref_width,
                    height=self._shared.image_ref_height,
                    aspect_ratio=aspect_ratio,
                    crop_size=[],
                    is_rand=False,
                    offset_func=None,
                )

                resize_result = await async_resize_image(params=resize_data)

                # Handle both tuple and direct image returns
                image = (
                    resize_result[0]
                    if isinstance(resize_result, tuple)
                    else resize_result
                )
            else:
                # Use original image if no aspect ratio is set
                image = pil_img

            # Convert to bytes
            self._obstacle_image = await self._pil_to_bytes(
                image,
                image_id=obstacle.get("label", "obstacle"),
            )

            return self._obstacle_image

        except HomeAssistantError as err:
            LOGGER.warning(
                "%s: Unexpected Error processing image: %r",
                self._file_name,
                err,
                exc_info=True,
            )
            await self._set_camera_mode(CameraModes.MAP_VIEW, "Error processing image")
            return None

    def get_obstacle_image(self) -> Optional[bytes]:
        """Get the current obstacle image."""
        return self._obstacle_image

    def is_processing(self) -> bool:
        """Check if obstacle view is currently processing."""
        return self._processing

    def clear_obstacle_image(self) -> None:
        """Clear the current obstacle image."""
        self._obstacle_image = None

    async def async_clear_cache(self) -> None:
        """Clear cached obstacle images from disk."""
        if self._prefetch_task and not self._prefetch_task.done():
            self._prefetch_task.cancel()
            self._prefetch_task = None

        if not self._obstacle_cache_dir.exists():
            return

        def _remove_cached_files() -> None:
            for file_path in self._obstacle_cache_dir.glob("*.jpg"):
                try:
                    file_path.unlink()
                except OSError as err:
                    LOGGER.warning(
                        "%s: Failed to delete cached obstacle image %s: %s",
                        self._file_name,
                        file_path,
                        err,
                        exc_info=True,
                    )

        await asyncio.to_thread(_remove_cached_files)
