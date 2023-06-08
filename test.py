import ai_basics
from PIL import Image

im = Image.open('edge_WWWWWWWWW_small_1.png')

imred = Image.fromarray(ai_basics.redden(im, 50))
imblu = Image.fromarray(ai_basics.blueify(im, 50))
imgreen = Image.fromarray(ai_basics.greenify(im, 50))

im.show('orig')
imred.show('bright')
imgreen.show('dark')
imblu.show('hicontrast')
