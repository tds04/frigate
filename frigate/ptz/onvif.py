"""Configure and control camera via onvif."""

import asyncio
import logging
import threading
import time
from enum import Enum
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import numpy
from onvif import ONVIFCamera, ONVIFError, ONVIFService
from zeep.exceptions import Fault, TransportError

from frigate.camera import PTZMetrics
from frigate.config import FrigateConfig, ZoomingModeEnum
from frigate.util.builtin import find_by_key

logger = logging.getLogger(__name__)


class OnvifCommandEnum(str, Enum):
    init = "init"
    move_down = "move_down"
    move_left = "move_left"
    move_relative = "move_relative"
    move_right = "move_right"
    move_up = "move_up"
    preset = "preset"
    stop = "stop"
    zoom_in = "zoom_in"
    zoom_out = "zoom_out"
    focus_in = "focus_in"
    focus_out = "focus_out"


class OnvifController:
    ptz_metrics: dict[str, PTZMetrics]

    def __init__(self, config: FrigateConfig, ptz_metrics: dict[str, PTZMetrics]) -> None:
        self.cams: dict[str, dict] = {}
        self.failed_cams: dict[str, dict] = {}
        self.max_retries = 5
        self.reset_timeout = 900
        self.config = config
        self.ptz_metrics = ptz_metrics
        self.status_locks: dict[str, asyncio.Lock] = {}

        self.loop = asyncio.new_event_loop()
        self.loop_thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self.loop_thread.start()

        self.camera_configs = {}
        for cam_name, cam in config.cameras.items():
            if cam.enabled and cam.onvif.host:
                self.camera_configs[cam_name] = cam
                self.status_locks[cam_name] = asyncio.Lock()

        asyncio.run_coroutine_threadsafe(self._init_cameras(), self.loop)

    def _run_event_loop(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    async def _init_cameras(self) -> None:
        for cam_name in self.camera_configs:
            await self._init_single_camera(cam_name)

    async def _init_single_camera(self, cam_name: str) -> bool:
        cam = self.camera_configs[cam_name]
        try:
            onvif = ONVIFCamera(
                cam.onvif.host,
                cam.onvif.port,
                cam.onvif.user,
                cam.onvif.password,
                wsdl_dir=str(Path(find_spec("onvif").origin).parent / "wsdl"),
                adjust_time=cam.onvif.ignore_time_mismatch,
                encrypt=not cam.onvif.tls_insecure,
            )

            self.cams[cam_name] = {
                "onvif": onvif,
                "init": False,
                "active": False,
                "features": [],
                "presets": {},
                "vendor": "unknown",
            }

            # Vendor detection (Axis-safe)
            try:
                dev = await onvif.create_devicemgmt_service()
                info = await dev.GetDeviceInformation()
                if info and "axis" in info.Manufacturer.lower():
                    self.cams[cam_name]["vendor"] = "axis"
                    logger.debug(f"{cam_name}: Detected Axis camera")
            except Exception:
                pass

            return True

        except Exception as e:
            logger.error(f"Failed to init ONVIF camera {cam_name}: {e}")
            self.failed_cams[cam_name] = {
                "retry_attempts": 0,
                "last_error": str(e),
                "last_attempt": time.time(),
            }
            return False

    async def _init_onvif(self, camera_name: str) -> bool:
        cam = self.cams[camera_name]
        onvif = cam["onvif"]

        await onvif.update_xaddrs()
        media = await onvif.create_media_service()
        profiles = await media.GetProfiles()

        profile = next(
            (
                p
                for p in profiles
                if p.VideoEncoderConfiguration
                and p.PTZConfiguration
                and (
                    p.PTZConfiguration.DefaultContinuousPanTiltVelocitySpace
                    or p.PTZConfiguration.DefaultContinuousZoomVelocitySpace
                )
            ),
            None,
        )

        if not profile:
            logger.error(f"No PTZ-capable profile for {camera_name}")
            return False

        ptz = await onvif.create_ptz_service()
        cam["ptz"] = ptz

        cam["move_request"] = ptz.create_type("ContinuousMove")
        cam["move_request"].ProfileToken = profile.token

        if self.config.cameras[camera_name].onvif.autotracking.enabled:
            request = ptz.create_type("GetConfigurationOptions")
            request.ConfigurationToken = profile.PTZConfiguration.token
            ptz_config = await ptz.GetConfigurationOptions(request)

            fov_space = next(
                (
                    s
                    for s in ptz_config.Spaces.RelativePanTiltTranslationSpace
                    if "TranslationSpaceFov" in s.URI
                ),
                None,
            )

            rel = ptz.create_type("RelativeMove")
            rel.ProfileToken = profile.token
            rel.Translation = {
                "PanTilt": {
                    "x": 0.0,
                    "y": 0.0,
                    "space": fov_space.URI if fov_space else None,
                }
            }

            cam["relative_fov_range"] = fov_space
            cam["relative_move_request"] = rel

            if cam["vendor"] == "axis":
                cam["zoom_limits"] = profile.PTZConfiguration.ZoomLimits

        cam["features"] = ["pt", "pt-r-fov", "zoom-a"]
        cam["init"] = True
        return True

    async def _move_relative(self, camera_name: str, pan, tilt, zoom, speed) -> None:
        cam = self.cams[camera_name]

        pan = numpy.clip(pan, -1.0, 1.0)
        tilt = numpy.clip(tilt, -1.0, 1.0)

        pan = numpy.interp(
            pan,
            [-1, 1],
            [
                cam["relative_fov_range"].XRange.Min,
                cam["relative_fov_range"].XRange.Max,
            ],
        )
        tilt = numpy.interp(
            tilt,
            [-1, 1],
            [
                cam["relative_fov_range"].YRange.Min,
                cam["relative_fov_range"].YRange.Max,
            ],
        )

        req = cam["relative_move_request"]
        req.Translation["PanTilt"]["x"] = pan
        req.Translation["PanTilt"]["y"] = tilt
        req.Speed = {"PanTilt": {"x": speed, "y": speed}}

        await cam["ptz"].RelativeMove(req)

    async def _zoom_absolute(self, camera_name: str, zoom, speed) -> None:
        cam = self.cams[camera_name]
        req = cam["ptz"].create_type("AbsoluteMove")
        req.ProfileToken = cam["move_request"].ProfileToken

        if cam["vendor"] == "axis":
            zoom = numpy.interp(
                zoom,
                [0, 1],
                [
                    cam["zoom_limits"]["Range"]["XRange"]["Min"],
                    cam["zoom_limits"]["Range"]["XRange"]["Max"],
                ],
            )

        req.Position = {"Zoom": zoom}
        req.Speed = {"Zoom": speed}

        await cam["ptz"].AbsoluteMove(req)

    async def handle_command_async(
        self, camera_name: str, command: OnvifCommandEnum, param: str = ""
    ) -> None:
        if not self.cams[camera_name]["init"]:
            if not await self._init_onvif(camera_name):
                return

        if command == OnvifCommandEnum.move_relative:
            _, pan, tilt = param.split("_")
            await self._move_relative(camera_name, float(pan), float(tilt), 0, 0.3)

    def handle_command(
        self, camera_name: str, command: OnvifCommandEnum, param: str = ""
    ) -> None:
        asyncio.run_coroutine_threadsafe(
            self.handle_command_async(camera_name, command, param), self.loop
        )

    def close(self) -> None:
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.loop_thread.join()

