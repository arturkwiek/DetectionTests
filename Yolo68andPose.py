"""
YOLO: detekcja obiektów + detekcja pozycji (pose) + 68 punktów twarzy (dlib)
+ nakładanie obrazków z lokalnego folderu - na żywo z kamery USB.

Instalacja:
    pip install ultralytics opencv-python dlib-bin
    (zamiast dlib-bin można użyć "pip install dlib" - wymaga CMake i kompilatora C++)

Model 68 punktów twarzy (shape_predictor_68_face_landmarks.dat) pobiera się
automatycznie przy pierwszym uruchomieniu (~100 MB po rozpakowaniu).

Nakładki:
    Wrzuć obrazki do folderu "nakladki" obok skryptu (PNG, JPG, JPEG, WEBP, BMP).
    PNG z przezroczystym tłem daje najlepszy efekt (np. okulary, maska, logo, ramka).
    Jeśli folder jest pusty, skrypt utworzy przykładowe okulary "okulary_przyklad.png".

Uruchomienie:
    python yolo_kamera_obiekty_pozycja.py                       # kamera 0
    python yolo_kamera_obiekty_pozycja.py --kamera 1            # inna kamera USB
    python yolo_kamera_obiekty_pozycja.py --nakladki C:\\moje\\obrazki
    python yolo_kamera_obiekty_pozycja.py --nakladka-skala 1.2 --nakladka-przesun-y -0.1

Klawisze w oknie podglądu:
    q / Esc  - wyjście
    s        - zapis bieżącej klatki (PNG) do folderu "zrzuty"
    o        - włącz/wyłącz detekcję obiektów
    p        - włącz/wyłącz detekcję pozycji
    f        - włącz/wyłącz 68 punktów twarzy
    n        - włącz/wyłącz numery punktów twarzy
    i        - włącz/wyłącz nakładkę (obrazek)
    [ / ]    - poprzedni / następny obrazek z folderu
    m        - tryb nakładki: twarz -> róg -> cały kadr
    r        - ponowne wczytanie folderu z nakładkami
"""

import argparse
import bz2
import math
import platform
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

try:
    import dlib
except ImportError:
    dlib = None

URL_MODELU_TWARZY = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
PLIK_MODELU_TWARZY = Path("shape_predictor_68_face_landmarks.dat")
ROZSZERZENIA_OBRAZOW = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
TRYBY_NAKLADKI = ["twarz", "rog", "kadr"]

# Grupy punktów twarzy w schemacie 68 punktów: (zakres, czy_zamknięty, kolor BGR)
GRUPY_TWARZY = [
    (range(0, 17), False, (255, 200, 0)),    # linia żuchwy
    (range(17, 22), False, (0, 200, 255)),   # prawa brew
    (range(22, 27), False, (0, 200, 255)),   # lewa brew
    (range(27, 31), False, (0, 255, 0)),     # grzbiet nosa
    (range(31, 36), False, (0, 255, 0)),     # dół nosa
    (range(36, 42), True, (255, 0, 255)),    # prawe oko
    (range(42, 48), True, (255, 0, 255)),    # lewe oko
    (range(48, 60), True, (0, 0, 255)),      # usta - zewnętrzne
    (range(60, 68), True, (0, 128, 255)),    # usta - wewnętrzne
]


def parse_args():
    ap = argparse.ArgumentParser(description="YOLO obiekty + pozycja + twarz 68 + nakładki z kamery USB")
    ap.add_argument("--kamera", type=int, default=0, help="indeks kamery USB (0, 1, 2...)")
    ap.add_argument("--model-obiekty", default="yolo11n.pt", help="model detekcji obiektów")
    ap.add_argument("--model-pozycja", default="yolo11n-pose.pt", help="model detekcji pozycji")
    ap.add_argument("--conf", type=float, default=0.35, help="próg pewności YOLO (0-1)")
    ap.add_argument("--szer", type=int, default=1280, help="szerokość obrazu z kamery")
    ap.add_argument("--wys", type=int, default=720, help="wysokość obrazu z kamery")
    ap.add_argument("--urzadzenie", default=None,
                    help="'cpu', '0' (GPU CUDA) lub 'mps' (Mac); domyślnie wybór automatyczny")
    ap.add_argument("--skala-twarzy", type=float, default=0.5,
                    help="skala obrazu do wyszukiwania twarzy (mniej = szybciej, ale gorzej dla małych twarzy)")
    ap.add_argument("--nakladki", default="nakladki", help="folder z obrazkami do nakładania")
    ap.add_argument("--nakladka-tryb", choices=TRYBY_NAKLADKI, default="twarz",
                    help="twarz = dopasowanie do oczu, rog = prawy górny róg, kadr = cały obraz")
    ap.add_argument("--nakladka-skala", type=float, default=1.0,
                    help="rozmiar nakładki: w trybie 'twarz' względem szerokości twarzy, "
                         "w trybie 'rog' względem 1/4 szerokości kadru")
    ap.add_argument("--nakladka-przesun-y", type=float, default=0.0,
                    help="przesunięcie nakładki na twarzy w pionie (ułamek szerokości twarzy, "
                         "ujemne = wyżej, np. -0.5 dla czapki, 0.4 dla wąsów)")
    ap.add_argument("--nakladka-przezroczystosc", type=float, default=1.0,
                    help="krycie nakładki 0-1 (1 = pełne)")
    return ap.parse_args()


# ---------------------------------------------------------------- kamera

def otworz_kamere(indeks, szer, wys):
    # Na Windows DirectShow otwiera kamery USB szybciej i stabilniej
    backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
    cap = cv2.VideoCapture(indeks, backend)
    if not cap.isOpened():
        cap = cv2.VideoCapture(indeks)  # próba z domyślnym backendem
    if not cap.isOpened():
        raise RuntimeError(f"Nie można otworzyć kamery o indeksie {indeks}. "
                           f"Sprawdź podłączenie lub spróbuj --kamera 1.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, szer)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, wys)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # mniejsze opóźnienie obrazu
    return cap


# ---------------------------------------------------------------- twarz 68

def przygotuj_model_twarzy():
    """Zwraca (detektor_twarzy, predyktor_68) albo (None, None), jeśli dlib niedostępny."""
    if dlib is None:
        print("UWAGA: brak biblioteki dlib - tryb 68 punktów twarzy wyłączony.\n"
              "       Zainstaluj: pip install dlib-bin")
        return None, None
    if not PLIK_MODELU_TWARZY.exists():
        print("Pobieranie modelu 68 punktów twarzy (jednorazowo, ~60 MB)...")
        plik_bz2 = PLIK_MODELU_TWARZY.with_suffix(".dat.bz2")
        urllib.request.urlretrieve(URL_MODELU_TWARZY, plik_bz2)
        with bz2.open(plik_bz2, "rb") as src, open(PLIK_MODELU_TWARZY, "wb") as dst:
            dst.write(src.read())
        plik_bz2.unlink()
        print(f"Zapisano model: {PLIK_MODELU_TWARZY}")
    return dlib.get_frontal_face_detector(), dlib.shape_predictor(str(PLIK_MODELU_TWARZY))


def wykryj_punkty_twarzy(klatka, detektor, predyktor, skala):
    """Zwraca listę twarzy; każda twarz to (prostokąt, lista 68 punktów (x, y))."""
    szara = cv2.cvtColor(klatka, cv2.COLOR_BGR2GRAY)
    mala = cv2.resize(szara, None, fx=skala, fy=skala) if skala != 1.0 else szara
    twarze = []
    for r in detektor(mala, 0):
        # przeskalowanie prostokąta z powrotem do pełnej rozdzielczości
        rect = dlib.rectangle(int(r.left() / skala), int(r.top() / skala),
                              int(r.right() / skala), int(r.bottom() / skala))
        ksztalt = predyktor(szara, rect)
        punkty = [(ksztalt.part(i).x, ksztalt.part(i).y) for i in range(68)]
        twarze.append((rect, punkty))
    return twarze


def rysuj_twarz(obraz, rect, punkty, numery=False):
    cv2.rectangle(obraz, (rect.left(), rect.top()), (rect.right(), rect.bottom()),
                  (200, 200, 200), 1)
    for zakres, zamkniety, kolor in GRUPY_TWARZY:
        idx = list(zakres)
        for a, b in zip(idx, idx[1:]):
            cv2.line(obraz, punkty[a], punkty[b], kolor, 1, cv2.LINE_AA)
        if zamkniety:
            cv2.line(obraz, punkty[idx[-1]], punkty[idx[0]], kolor, 1, cv2.LINE_AA)
        for i in idx:
            cv2.circle(obraz, punkty[i], 2, kolor, -1, cv2.LINE_AA)
            if numery:
                cv2.putText(obraz, str(i), (punkty[i][0] + 3, punkty[i][1] - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)


# ---------------------------------------------------------------- nakładki

def utworz_przykladowa_nakladke(sciezka):
    """Rysuje proste okulary na przezroczystym tle (PNG z kanałem alfa)."""
    img = np.zeros((200, 600, 4), dtype=np.uint8)
    czarny = (20, 20, 20, 255)
    for cx in (150, 450):
        cv2.ellipse(img, (cx, 100), (110, 80), 0, 0, 360, (60, 40, 20, 140), -1)  # szkło
        cv2.ellipse(img, (cx, 100), (110, 80), 0, 0, 360, czarny, 14, cv2.LINE_AA)  # oprawka
    cv2.line(img, (258, 90), (342, 90), czarny, 14, cv2.LINE_AA)  # mostek
    cv2.imwrite(str(sciezka), img)


def wczytaj_nakladki(folder):
    """Wczytuje obrazki z folderu jako BGRA. Zwraca listę (nazwa, obraz)."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    pliki = sorted(p for p in folder.iterdir() if p.suffix.lower() in ROZSZERZENIA_OBRAZOW)
    if not pliki:
        przyklad = folder / "okulary_przyklad.png"
        utworz_przykladowa_nakladke(przyklad)
        print(f"Folder nakładek był pusty - utworzono przykład: {przyklad}")
        pliki = [przyklad]

    nakladki = []
    for p in pliki:
        # imdecode zamiast imread - obsługuje polskie znaki w ścieżkach na Windows
        dane = np.fromfile(str(p), dtype=np.uint8)
        img = cv2.imdecode(dane, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Pominięto (nie da się odczytać): {p.name}")
            continue
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)  # brak alfy = pełne krycie
        nakladki.append((p.name, img))
    print(f"Wczytano nakładek: {len(nakladki)} z folderu {folder.resolve()}")
    return nakladki


def obroc_z_alfa(img, kat_stopnie):
    """Obraca obraz BGRA o kąt (stopnie, przeciwnie do wskazówek zegara) bez przycinania."""
    if abs(kat_stopnie) < 0.5:
        return img
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), kat_stopnie, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(img, M, (nw, nh), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))


def naloz(obraz, nakladka, cx, cy, szerokosc, kat=0.0, krycie=1.0):
    """Nakłada obraz BGRA na klatkę BGR, wyśrodkowany w (cx, cy), z mieszaniem alfa.
    Fragmenty wychodzące poza kadr są przycinane."""
    if szerokosc < 2:
        return
    h0, w0 = nakladka.shape[:2]
    nowa_w = int(szerokosc)
    nowa_h = max(1, int(h0 * szerokosc / w0))
    img = cv2.resize(nakladka, (nowa_w, nowa_h), interpolation=cv2.INTER_AREA)
    img = obroc_z_alfa(img, kat)

    h, w = img.shape[:2]
    x0, y0 = int(cx - w / 2), int(cy - h / 2)
    H, W = obraz.shape[:2]
    # część wspólna nakładki i kadru
    ox0, oy0 = max(x0, 0), max(y0, 0)
    ox1, oy1 = min(x0 + w, W), min(y0 + h, H)
    if ox0 >= ox1 or oy0 >= oy1:
        return
    wycinek = img[oy0 - y0:oy1 - y0, ox0 - x0:ox1 - x0]
    alfa = (wycinek[:, :, 3:4].astype(np.float32) / 255.0) * krycie
    tlo = obraz[oy0:oy1, ox0:ox1].astype(np.float32)
    obraz[oy0:oy1, ox0:ox1] = (alfa * wycinek[:, :, :3] + (1 - alfa) * tlo).astype(np.uint8)


def naloz_na_twarz(obraz, punkty, nakladka, skala, przesun_y, krycie):
    """Dopasowuje nakładkę do twarzy: środek między oczami, obrót wg linii oczu,
    szerokość = szerokość twarzy (punkty 0-16) * skala."""
    oko_p = np.mean(punkty[36:42], axis=0)
    oko_l = np.mean(punkty[42:48], axis=0)
    dx, dy = oko_l[0] - oko_p[0], oko_l[1] - oko_p[1]
    kat = -math.degrees(math.atan2(dy, dx))  # OpenCV: dodatni kąt = przeciwnie do zegara
    szer_twarzy = math.dist(punkty[0], punkty[16])
    cx, cy = (oko_p + oko_l) / 2
    # przesunięcie w pionie wzdłuż osi twarzy (uwzględnia przechylenie głowy)
    rad = math.atan2(dy, dx)
    przes = przesun_y * szer_twarzy
    cx += -math.sin(rad) * przes
    cy += math.cos(rad) * przes
    naloz(obraz, nakladka, cx, cy, szer_twarzy * skala, kat, krycie)


# ---------------------------------------------------------------- pętla główna

def main():
    args = parse_args()

    print("Ładowanie modeli YOLO (przy pierwszym uruchomieniu zostaną pobrane)...")
    model_obj = YOLO(args.model_obiekty)
    model_pose = YOLO(args.model_pozycja)
    detektor_twarzy, predyktor_twarzy = przygotuj_model_twarzy()
    nakladki = wczytaj_nakladki(args.nakladki)

    cap = otworz_kamere(args.kamera, args.szer, args.wys)
    print(f"Kamera {args.kamera}: "
          f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

    folder_zrzutow = Path("zrzuty")
    pokaz_obiekty, pokaz_pozycje = True, True
    pokaz_twarz = predyktor_twarzy is not None
    pokaz_numery = False
    pokaz_nakladke = bool(nakladki)
    nr_nakladki = 0
    tryb = args.nakladka_tryb
    if tryb == "twarz" and predyktor_twarzy is None:
        tryb = "rog"  # bez dlib nie ma punktów twarzy
    fps, t_prev = 0.0, time.time()
    okno = "YOLO - obiekty + pozycja + twarz 68 + nakladki (q = wyjscie)"

    try:
        while True:
            ok, klatka = cap.read()
            if not ok:
                print("Brak klatki z kamery - kończę.")
                break

            obraz = klatka.copy()
            H, W = obraz.shape[:2]
            n_obiektow = n_osob = n_twarzy = 0
            wspolne = dict(conf=args.conf, verbose=False)
            if args.urzadzenie is not None:
                wspolne["device"] = args.urzadzenie

            # 1) Detekcja obiektów (80 klas COCO: osoba, auto, kubek, telefon...)
            if pokaz_obiekty:
                wyn_obj = model_obj(klatka, **wspolne)[0]
                n_obiektow = len(wyn_obj.boxes)
                obraz = wyn_obj.plot(img=obraz)

            # 2) Detekcja pozycji (17 punktów szkieletu na każdą osobę)
            if pokaz_pozycje:
                wyn_pose = model_pose(klatka, **wspolne)[0]
                n_osob = len(wyn_pose.boxes)
                # gdy ramki rysuje już model obiektów, rysujemy tylko szkielety
                obraz = wyn_pose.plot(img=obraz, boxes=not pokaz_obiekty)

            # 3) 68 punktów twarzy - liczone, gdy trzeba je pokazać albo nałożyć obrazek na twarz
            twarze = []
            potrzebne_twarze = pokaz_twarz or (pokaz_nakladke and tryb == "twarz")
            if potrzebne_twarze and predyktor_twarzy is not None:
                twarze = wykryj_punkty_twarzy(klatka, detektor_twarzy, predyktor_twarzy,
                                              args.skala_twarzy)
                n_twarzy = len(twarze)

            # 4) Nakładka z folderu
            if pokaz_nakladke and nakladki:
                nazwa, img = nakladki[nr_nakladki]
                if tryb == "twarz":
                    for _, punkty in twarze:
                        naloz_na_twarz(obraz, punkty, img, args.nakladka_skala,
                                       args.nakladka_przesun_y, args.nakladka_przezroczystosc)
                elif tryb == "rog":
                    szer = W * 0.25 * args.nakladka_skala
                    wys = img.shape[0] * szer / img.shape[1]
                    naloz(obraz, img, W - 15 - szer / 2, 40 + wys / 2, szer,
                          krycie=args.nakladka_przezroczystosc)
                else:  # kadr - cały obraz (np. ramka, filtr, maska kolorystyczna)
                    pelna = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
                    naloz(obraz, pelna, W / 2, H / 2, W, krycie=args.nakladka_przezroczystosc)

            # punkty twarzy rysujemy na wierzchu, żeby nakładka ich nie zasłoniła
            if pokaz_twarz:
                for rect, punkty in twarze:
                    rysuj_twarz(obraz, rect, punkty, pokaz_numery)

            # FPS (wygładzony)
            teraz = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(teraz - t_prev, 1e-6))
            t_prev = teraz

            info = (f"FPS: {fps:4.1f} | obiekty: {n_obiektow} | osoby: {n_osob} | twarze: {n_twarzy} | "
                    f"[o]{'ON' if pokaz_obiekty else 'OFF'} "
                    f"[p]{'ON' if pokaz_pozycje else 'OFF'} "
                    f"[f]{'ON' if pokaz_twarz else 'OFF'} "
                    f"[i]{'ON' if pokaz_nakladke else 'OFF'}")
            cv2.rectangle(obraz, (0, 0), (W, 30), (0, 0, 0), -1)
            cv2.putText(obraz, info, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 1, cv2.LINE_AA)
            if nakladki:
                opis = (f"Nakladka {nr_nakladki + 1}/{len(nakladki)}: {nakladki[nr_nakladki][0]} | "
                        f"tryb [m]: {tryb} | [ ] zmiana | [r] przeladuj")
                cv2.rectangle(obraz, (0, H - 28), (W, H), (0, 0, 0), -1)
                cv2.putText(obraz, opis, (10, H - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow(okno, obraz)
            klawisz = cv2.waitKey(1) & 0xFF
            if klawisz in (ord("q"), 27):
                break
            elif klawisz == ord("s"):
                folder_zrzutow.mkdir(exist_ok=True)
                plik = folder_zrzutow / f"zrzut_{time.strftime('%Y%m%d_%H%M%S')}.png"
                cv2.imwrite(str(plik), obraz)
                print(f"Zapisano: {plik}")
            elif klawisz == ord("o"):
                pokaz_obiekty = not pokaz_obiekty
            elif klawisz == ord("p"):
                pokaz_pozycje = not pokaz_pozycje
            elif klawisz == ord("f"):
                if predyktor_twarzy is None:
                    print("Tryb twarzy niedostępny - zainstaluj dlib (pip install dlib-bin).")
                else:
                    pokaz_twarz = not pokaz_twarz
            elif klawisz == ord("n"):
                pokaz_numery = not pokaz_numery
            elif klawisz == ord("i"):
                pokaz_nakladke = not pokaz_nakladke and bool(nakladki)
            elif klawisz == ord("]") and nakladki:
                nr_nakladki = (nr_nakladki + 1) % len(nakladki)
            elif klawisz == ord("[") and nakladki:
                nr_nakladki = (nr_nakladki - 1) % len(nakladki)
            elif klawisz == ord("m"):
                dostepne = TRYBY_NAKLADKI if predyktor_twarzy is not None else ["rog", "kadr"]
                tryb = dostepne[(dostepne.index(tryb) + 1) % len(dostepne)] if tryb in dostepne else dostepne[0]
            elif klawisz == ord("r"):
                nakladki = wczytaj_nakladki(args.nakladki)
                nr_nakladki = 0
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
