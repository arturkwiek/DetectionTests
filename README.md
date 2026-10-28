# DetectionTests

Podgląd na żywo z kamery: detekcja obiektów i pozycji (YOLO11n), 68 punktów twarzy (dlib)
oraz nakładanie obrazków z folderu `nakladki` (okulary, maski, ramki).

| Plik | Do czego |
|------|----------|
| `Yolo68andPose.py` | główna aplikacja |
| `strumien_kamery.py` | strumień MJPEG z kamery pod Windowsem - źródło obrazu dla WSL |

Modele (`yolo11n.pt`, `yolo11n-pose.pt`, `shape_predictor_68_face_landmarks.dat`)
pobierają się same przy pierwszym uruchomieniu.

## Instalacja

**Windows** (tylko CPU):

```powershell
py -3.10 -m venv .venv-win
.\.venv-win\Scripts\python.exe -m pip install ultralytics opencv-python dlib-bin
```

**WSL / Linux** (GPU przez CUDA):

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Quick guide - źródło obrazu

### 1. Windows, kamera lokalna

```powershell
.\.venv-win\Scripts\python.exe Yolo68andPose.py --kamera 0   # zwykle kamera zintegrowana
.\.venv-win\Scripts\python.exe Yolo68andPose.py --kamera 1   # zwykle kamera USB
```

Numeracja zależy od kolejności podłączenia - jeśli obraz jest czarny, spróbuj innego indeksu.

### 2. WSL, kamera ze strumienia Windows (zalecane pod WSL)

Kamery przekazane do WSL przez `usbipd` otwierają się, ale nie przesyłają klatek,
dlatego kamera zostaje pod Windowsem, a obraz idzie przez HTTP.

Windows (osobne okno, zostaw włączone):

```powershell
.\.venv-win\Scripts\python.exe strumien_kamery.py --kamera 1              # USB -> port 8080
.\.venv-win\Scripts\python.exe strumien_kamery.py --kamera 0 --port 8081  # druga kamera równolegle
```

WSL:

```bash
HOST=$(ip route show default | awk '{print $3}')    # IP Windowsa widziane z WSL, np. 172.22.128.1
.venv/bin/python Yolo68andPose.py --kamera http://$HOST:8080/
```

Jeśli IP hosta jest inne niż `172.22.128.1`, podaj je też serwerowi: `strumien_kamery.py --adres <IP>`.
Serwer nasłuchuje tylko na interfejsie WSL, więc obraz nie jest widoczny w sieci lokalnej.
Przy pierwszym uruchomieniu zapora Windows może zapytać o dostęp - zezwól dla sieci prywatnej.

Opcje `strumien_kamery.py`: `--kamera`, `--adres`, `--port`, `--szer`/`--wys` (domyślnie 640x480),
`--jakosc` (JPEG 1-100, domyślnie 85).

### 3. Inne strumienie

`--kamera` przyjmuje dowolny adres obsługiwany przez OpenCV/FFmpeg, np. kamerę IP lub plik wideo:

```bash
.venv/bin/python Yolo68andPose.py --kamera rtsp://192.168.1.50:554/stream
.venv/bin/python Yolo68andPose.py --kamera nagranie.mp4
```

## Quick guide - opcje uruchamiania

```bash
# wydajność
python Yolo68andPose.py --urzadzenie 0          # GPU CUDA (cpu / 0 / mps)
python Yolo68andPose.py --szer 640 --wys 480     # niższa rozdzielczość kamery
python Yolo68andPose.py --skala-twarzy 0.3       # szybsze szukanie twarzy (gorzej dla małych)
python Yolo68andPose.py --conf 0.5               # mniej, ale pewniejszych detekcji

# inne modele YOLO
python Yolo68andPose.py --model-obiekty yolo11s.pt --model-pozycja yolo11s-pose.pt

# nakładki
python Yolo68andPose.py --nakladki C:\moje\obrazki               # inny folder z obrazkami
python Yolo68andPose.py --nakladka-tryb rog                       # twarz / rog / kadr
python Yolo68andPose.py --nakladka-skala 1.2 --nakladka-przesun-y -0.5   # czapka
python Yolo68andPose.py --nakladka-przesun-y 0.4                  # wąsy
python Yolo68andPose.py --nakladka-przezroczystosc 0.6
```

Nakładki: PNG z przezroczystym tłem daje najlepszy efekt (obsługiwane także JPG, WEBP, BMP).

## Klawisze w oknie podglądu

| Klawisz | Akcja |
|---------|-------|
| `q` / `Esc` | wyjście |
| `s` | zrzut klatki do `zrzuty/` |
| `o` | detekcja obiektów on/off |
| `p` | detekcja pozycji on/off |
| `f` | 68 punktów twarzy on/off |
| `n` | numery punktów twarzy |
| `i` | nakładka on/off |
| `[` / `]` | poprzedni / następny obrazek |
| `m` | tryb nakładki: twarz → róg → kadr |
| `r` | ponowne wczytanie folderu nakładek |
