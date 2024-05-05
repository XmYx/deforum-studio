from qtpy.QtGui import QPixmap, QImage


def npArrayToQPixmap(arr):
    """Convert a numpy array to QPixmap."""
    height, width, _ = arr.shape
    bytes_per_line = 3 * width
    qImg = QImage(
        arr.data,
        width,
        height,
        bytes_per_line,
        QImage.Format.Format_RGB888,
    )
    # Convert QImage to QPixmap
    pixmap = QPixmap.fromImage(qImg)
    return pixmap
