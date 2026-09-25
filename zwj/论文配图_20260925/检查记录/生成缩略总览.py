from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "检查记录"
FONT = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 14)


def make_sheet(folders: list[Path], filename: str) -> None:
    files = []
    for folder in folders:
        files.extend(sorted(p for p in folder.glob("*.png") if not p.stem.endswith("_grayscale")))
    columns = 4
    cell_w, image_h, label_h, margin = 380, 250, 32, 14
    rows = (len(files) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * cell_w + margin * 2, rows * (image_h + label_h) + margin * 2), "white")
    draw = ImageDraw.Draw(canvas)
    for index, path in enumerate(files):
        x = margin + (index % columns) * cell_w
        y = margin + (index // columns) * (image_h + label_h)
        with Image.open(path) as original:
            image = original.convert("RGB")
            image.thumbnail((cell_w - 12, image_h - 12), Image.Resampling.LANCZOS)
            canvas.paste(image, (x + (cell_w - image.width) // 2, y + (image_h - image.height) // 2))
        draw.rectangle((x, y, x + cell_w - 1, y + image_h + label_h), outline="#D0D5DD", width=1)
        label = f"{path.parent.name} / {path.stem}"
        draw.text((x + 6, y + image_h + 7), label, fill="#222222", font=FONT)
    target = OUT / filename
    canvas.save(target, dpi=(150, 150), optimize=True)
    print(f"{target}: {len(files)} 张图")


if __name__ == "__main__":
    paper = ROOT / "论文版"
    plot = ROOT / "绘图版"
    make_sheet(
        [plot / "核心图表", plot / "Q2补充图表", plot / "Q3补充图表"],
        "绘图版图缩览.png",
    )
    make_sheet([paper / "核心图表"], "论文版核心图缩览.png")
    make_sheet([paper / "Q2补充图表", paper / "Q3补充图表"], "论文版补充图缩览.png")
