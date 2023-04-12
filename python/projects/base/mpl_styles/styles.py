from pathlib import Path

import matplotlib.pyplot as plt

mpl_styles_path = Path(__file__).parent
publication_style_names = ("default", "publication", "tex")

def get_style(name):
    return mpl_styles_path / f"{name}.mplstyle"

def use_styles(*styles):
    print("Using plotting styles")
    use_styles = []
    for style in styles:
        if style == "default" or style in plt.style.available:
            use_styles.append(style)
        else:
            use_styles.append(get_style(style))
        print(f"  {use_styles[-1]}")
    plt.style.use(use_styles)

def set_publication_styles():
    use_styles(*publication_style_names)

def set_dark_styles():
    use_styles("dark_background", "publication", "tex")
