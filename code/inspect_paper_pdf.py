"""Render a manuscript PDF and build a contact sheet for layout inspection."""
from pathlib import Path
import sys
CODE_ROOT=Path(__file__).resolve().parent
PROJECT_ROOT=CODE_ROOT.parent
sys.path.insert(0,str(PROJECT_ROOT/"python_packages"/"figure"))
import fitz
from PIL import Image,ImageDraw

if len(sys.argv) <= 1:
    raise SystemExit("usage: python inspect_paper_pdf.py <manuscript.pdf>")
src=Path(sys.argv[1]).resolve()
out=PROJECT_ROOT/"record"/"paper"/"tmp"/(src.stem.lower()+"_render")
out.mkdir(parents=True,exist_ok=True)
doc=fitz.open(src); pages=[]
for i,p in enumerate(doc):
    pix=p.get_pixmap(dpi=130,alpha=False); path=out/f"page_{i+1:02d}.png";pix.save(path)
    im=Image.open(path).convert("RGB");im.thumbnail((510,660));pages.append(im.copy())
cols=3; rows=(len(pages)+cols-1)//cols
sheet=Image.new("RGB",(cols*530,rows*700),(225,225,225));draw=ImageDraw.Draw(sheet)
for i,im in enumerate(pages):
    x=(i%cols)*530+10;y=(i//cols)*700+28;sheet.paste(im,(x,y));draw.text((x,7+(i//cols)*700),f"Page {i+1}",fill="black")
sheet.save(out/"contact_sheet.png")
print(f"pages={len(doc)} contact={out/'contact_sheet.png'}")
