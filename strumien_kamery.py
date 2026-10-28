"""
Strumień MJPEG z kamery po stronie Windows - dla aplikacji uruchamianej pod WSL.

Kamery USB przekazane do WSL przez usbipd często nie przesyłają klatek, więc kamera
zostaje pod Windowsem, a obraz idzie do WSL przez HTTP.

Uruchomienie (Windows):
    python strumien_kamery.py                  # kamera 1, port 8080
    python strumien_kamery.py --kamera 0 --port 8081

Odbiór (WSL) - IP hosta Windows z punktu widzenia WSL:
    HOST=$(ip route show default | awk '{print $3}')
    python Yolo68andPose.py --kamera http://$HOST:8080/
"""

import argparse
import platform
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

GRANICA = b"klatka"


def parse_args():
    ap = argparse.ArgumentParser(description="Strumień MJPEG z kamery przez HTTP")
    ap.add_argument("--kamera", type=int, default=1, help="indeks kamery (Windows)")
    ap.add_argument("--adres", default="172.22.128.1",
                    help="adres nasłuchu - domyślnie interfejs WSL po stronie Windows, żeby obraz "
                         "nie był widoczny w sieci lokalnej (w WSL: ip route show default)")
    ap.add_argument("--port", type=int, default=8080, help="port HTTP")
    ap.add_argument("--szer", type=int, default=640, help="szerokość obrazu z kamery")
    ap.add_argument("--wys", type=int, default=480, help="wysokość obrazu z kamery")
    ap.add_argument("--jakosc", type=int, default=85, help="jakość JPEG 1-100")
    return ap.parse_args()


class Kamera:
    """Czyta kamerę w osobnym wątku i trzyma tylko najnowszą klatkę (JPEG),
    żeby odbiorca nie dostawał zaległych klatek."""

    def __init__(self, indeks, szer, wys, jakosc):
        backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(indeks, backend)
        if not self.cap.isOpened():
            raise RuntimeError(f"Nie można otworzyć kamery o indeksie {indeks}.")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, szer)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, wys)
        self.jakosc = [cv2.IMWRITE_JPEG_QUALITY, jakosc]
        self.jpeg, self.nr = None, 0
        self.warunek = threading.Condition()
        threading.Thread(target=self._petla, daemon=True).start()

    def _petla(self):
        while True:
            ok, klatka = self.cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            ok, bufor = cv2.imencode(".jpg", klatka, self.jakosc)
            if ok:
                with self.warunek:
                    self.jpeg, self.nr = bufor.tobytes(), self.nr + 1
                    self.warunek.notify_all()

    def nastepna(self, ostatni_nr):
        """Czeka na klatkę nowszą niż ostatni_nr. Zwraca (nr, jpeg)."""
        with self.warunek:
            self.warunek.wait_for(lambda: self.nr != ostatni_nr, timeout=5)
            return self.nr, self.jpeg


def utworz_handler(kamera):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={GRANICA.decode()}")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            nr = 0
            try:
                while True:
                    nr, jpeg = kamera.nastepna(nr)
                    if jpeg is None:
                        continue
                    self.wfile.write(b"--" + GRANICA + b"\r\nContent-Type: image/jpeg\r\n"
                                     + f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
                                     + jpeg + b"\r\n")
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass  # odbiorca się rozłączył

        def log_message(self, fmt, *args):
            print(f"{self.client_address[0]} - {fmt % args}")

    return Handler


def main():
    args = parse_args()
    kamera = Kamera(args.kamera, args.szer, args.wys, args.jakosc)
    print(f"Kamera {args.kamera}: "
          f"{int(kamera.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(kamera.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    serwer = ThreadingHTTPServer((args.adres, args.port), utworz_handler(kamera))
    serwer.daemon_threads = True
    print(f"Strumień MJPEG: http://{args.adres}:{args.port}/  (Ctrl+C = wyjście)")
    try:
        serwer.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        serwer.server_close()
        kamera.cap.release()


if __name__ == "__main__":
    main()
