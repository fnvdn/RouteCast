"""Create a Python-backend QA contact sheet from manuscript PNG previews."""
from pathlib import Path
import sys
CODE_ROOT=Path(__file__).resolve().parent
PROJECT_ROOT=CODE_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT/"python_packages"/"figure"))
from PIL import Image, ImageDraw

root=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else PROJECT_ROOT/"record"/"paper"/"figures"
files=sorted(root.glob("Fig*.png"))+sorted(root.glob("ExtFig*.png"))
thumbs=[]
for p in files:
    im=Image.open(p).convert("RGB"); im.thumbnail((720,420))
    canvas=Image.new("RGB",(760,470),"white"); canvas.paste(im,((760-im.width)//2,35))
    ImageDraw.Draw(canvas).text((12,10),p.stem,fill="black")
    thumbs.append(canvas)
cols=2; rows=(len(thumbs)+cols-1)//cols
sheet=Image.new("RGB",(cols*760,rows*470),(235,235,235))
for i,im in enumerate(thumbs):sheet.paste(im,((i%cols)*760,(i//cols)*470))
sheet.save(root/"QA_contact_sheet.png",dpi=(150,150))
print(root/"QA_contact_sheet.png")
