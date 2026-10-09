from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QImage
from PySide6.QtMultimedia import QVideoFrame

from lerobot_studio.ui.video_surface import SoftwareVideoSurface


def test_video_surface_is_opaque_black_without_a_frame(qtbot):
    surface = SoftwareVideoSurface()
    surface.resize(320, 240)
    qtbot.addWidget(surface)
    surface.show()
    qtbot.waitExposed(surface)

    image = surface.grab().toImage()
    assert image.pixelColor(image.rect().center()) == QColor(Qt.GlobalColor.black)


def test_video_surface_paints_video_frame_inside_regular_widget(qtbot):
    surface = SoftwareVideoSurface()
    surface.resize(320, 240)
    qtbot.addWidget(surface)
    surface.show()
    qtbot.waitExposed(surface)
    source = QImage(64, 48, QImage.Format.Format_RGB32)
    source.fill(QColor(Qt.GlobalColor.red))

    surface.present(QVideoFrame(source))
    surface.repaint()

    image = surface.grab().toImage()
    assert image.pixelColor(image.rect().center()) == QColor(Qt.GlobalColor.red)
