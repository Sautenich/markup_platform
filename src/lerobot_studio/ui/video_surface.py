from __future__ import annotations

from PySide6.QtCore import QRect, Qt, Slot
from PySide6.QtGui import QImage, QPainter, QPaintEvent
from PySide6.QtMultimedia import QVideoFrame
from PySide6.QtWidgets import QWidget


class SoftwareVideoSurface(QWidget):
    """An opaque, backing-store-safe video canvas.

    QVideoWidget may create a native overlay surface.  On some Linux
    compositors that surface intermittently contains stale pixels copied from
    the application window.  This widget receives frames through QVideoSink
    and paints them as regular QImages in the GUI thread instead.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._image = QImage()
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setMinimumSize(160, 120)

    @Slot(QVideoFrame)
    def present(self, frame: QVideoFrame) -> None:
        image = frame.toImage()
        if image.isNull():
            return
        self._image = image
        self.update()

    def clear(self) -> None:
        self._image = QImage()
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802
        painter = QPainter(self)
        try:
            painter.fillRect(self.rect(), Qt.GlobalColor.black)
            if not self._image.isNull():
                size = self._image.size()
                size.scale(self.size(), Qt.AspectRatioMode.KeepAspectRatio)
                target = QRect(
                    (self.width() - size.width()) // 2,
                    (self.height() - size.height()) // 2,
                    size.width(),
                    size.height(),
                )
                painter.drawImage(target, self._image)
        finally:
            painter.end()
