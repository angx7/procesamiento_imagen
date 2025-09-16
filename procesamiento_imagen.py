import cv2
import numpy as np

# ============
# Configuración de rutas de las imágenes
# ============
IN_PATHS = {
    "frontal": "luz_frontal.jpg",
    "natural": "luz_natural.jpg",
    "superior": "luz_superior.jpg",
}

# Parámetros para redimensionar y cuantizar
RESIZE_WIDTH = 900
LEVELS = [256, 64, 32, 16, 8, 2]  # cuantización


# =================
# Utilidades
# =================


# Carga y redimensiona manteniendo la relación de aspecto
def load_and_resize(path, width=900):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"No se pudo abrir: {path}")
    h, w = img.shape[:2]
    scale = width / w
    return cv2.resize(
        img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
    )


# Convierte BGR a escala de grises
def to_gray(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


# Cuantiza una imagen en niveles dados
def quantize(gray, levels):
    step = max(1, 256 // levels)
    q = (gray // step) * step
    return q.astype(np.uint8)


# Binarización usando el método de Otsu
def otsu(gray):
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return bw


def components_colormap(binary, connectivity=4):
    # labels: 0 = fondo, 1..N = objetos
    num_labels, labels = cv2.connectedComponents(binary, connectivity=connectivity)
    if labels.max() == 0:  # todo fondo
        colored = np.full((*binary.shape, 3), 255, np.uint8)
        return colored, 0
    label_hue = np.uint8(179 * labels / labels.max())
    sat = np.full_like(label_hue, 255)
    val = np.full_like(label_hue, 255)
    hsv = cv2.merge([label_hue, sat, val])
    colored = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    colored[labels == 0] = (255, 255, 255)  # fondo blanco
    return colored, num_labels - 1


def banner(img, text):
    bar = np.full((50, img.shape[1], 3), 255, np.uint8)
    cv2.putText(
        bar, text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA
    )
    return np.vstack([bar, img])


def stack_rows(*imgs):
    # ajusta alturas a la mínima y apila horizontalmente
    h = min(i.shape[0] for i in imgs)
    rs = [cv2.resize(i, (int(i.shape[1] * h / i.shape[0]), h)) for i in imgs]
    return np.hstack(rs)


def show(title, img):
    # Redimensionar si la imagen es muy grande
    max_width = 1200
    max_height = 800
    h, w = img.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyWindow(title)


# ============
# Pipeline
# ============
def main():
    # Cargar imágenes
    imgs = {name: load_and_resize(p, RESIZE_WIDTH) for name, p in IN_PATHS.items()}
    ref = imgs.get("natural", next(iter(imgs.values())))

    # 1) Mostrar originales
    show(
        "Originales (presiona una tecla para continuar)",
        stack_rows(*[banner(imgs[n], n) for n in imgs]),
    )

    # 2) Grayscale + Cuantización
    for name, img in imgs.items():
        gray = to_gray(img)
        quant_imgs = [cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)]
        for L in LEVELS[1:]:
            quant_imgs.append(cv2.cvtColor(quantize(gray, L), cv2.COLOR_GRAY2BGR))
        panel = stack_rows(*quant_imgs)
        panel = banner(
            panel, f"{name} | Gris, Cuantización: {', '.join(map(str, LEVELS))} niveles"
        )
        show(f"Gris + Cuantización ({name})", panel)

    # 3) Binarización + Conectividades (4, 8 y complementaria)
    for name, img in imgs.items():
        gray = to_gray(img)
        bw = otsu(gray)

        c4_img, c4 = components_colormap(bw, connectivity=4)
        c8_img, c8 = components_colormap(bw, connectivity=8)

        # Complementaria: contamos componentes del fondo (invertimos)
        bw_inv = cv2.bitwise_not(bw)
        comp_img, comp = components_colormap(bw_inv, connectivity=8)

        c4_img = banner(c4_img, f"{name} | Conectividad 4: {c4} objetos")
        c8_img = banner(c8_img, f"{name} | Conectividad 8: {c8} objetos")
        comp_img = banner(
            comp_img, f"{name} | Componentes del fondo (complementaria): {comp}"
        )

        panel = stack_rows(c4_img, c8_img, comp_img)
        show(f"Conectividad ({name})", panel)

    # 4) Operadores aritméticos (por imagen contra 'natural')
    for name, img in imgs.items():
        sum_img = cv2.add(img, ref)
        diff_img = cv2.absdiff(img, ref)
        blend = cv2.addWeighted(img, 0.5, ref, 0.5, 0)
        bright = cv2.convertScaleAbs(img, alpha=1.25, beta=10)
        dark = cv2.convertScaleAbs(img, alpha=0.85, beta=-10)
        mult_img = cv2.multiply(img.astype(np.float32)/255, ref.astype(np.float32)/255)
        mult_img = np.clip(mult_img * 255, 0, 255).astype(np.uint8)

        a = banner(sum_img, f"{name} + natural (suma saturada)")
        b = banner(diff_img, f"|{name} - natural| (diferencia)")
        c = banner(blend, f"Mezcla 0.5/0.5 con natural")
        m = banner(mult_img, f"{name} * natural (multiplicación)")
        d = banner(bright, f"{name} escala alpha=1.25, beta=10")
        e = banner(dark, f"{name} escala alpha=0.85, beta=-10")

        # mostrar en dos tandas para no saturar
        show(f"Aritméticos 1/2 ({name})", stack_rows(a, b, c, m))
        show(f"Aritméticos 2/2 ({name})", stack_rows(d, e))

    # 5) Kernels (suavizado, realce, bordes)
    for name, img in imgs.items():
        gray = to_gray(img)

        blur = cv2.blur(img, (5, 5))
        gauss = cv2.GaussianBlur(img, (5, 5), 1.0)
        lap = cv2.Laplacian(gray, ddepth=cv2.CV_16S, ksize=3)
        lap = cv2.convertScaleAbs(lap)
        sobx = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
        soby = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
        sobx = cv2.convertScaleAbs(sobx)
        soby = cv2.convertScaleAbs(soby)

        k1 = banner(blur, f"{name} | Suavizado promedio (5x5)")
        k2 = banner(gauss, f"{name} | Gaussiano (5x5, sigma=1)")
        k3 = banner(cv2.cvtColor(lap, cv2.COLOR_GRAY2BGR), f"{name} | Laplaciano")
        k4 = banner(cv2.cvtColor(sobx, cv2.COLOR_GRAY2BGR), f"{name} | Sobel X")
        k5 = banner(cv2.cvtColor(soby, cv2.COLOR_GRAY2BGR), f"{name} | Sobel Y")

        show(f"Kernels 1/2 ({name})", stack_rows(k1, k2, k3))
        show(f"Kernels 2/2 ({name})", stack_rows(k4, k5))

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
