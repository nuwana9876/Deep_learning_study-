import sys
import cv2

def get_selective_search():
  # create selective search segmentation object using default parameters
  gs = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
  return gs

def config(gs, img, strategy = 'q'):
  # set input image on which we will run segmentation
  gs.setBaseImage(img)

  if (strategy == 's'):
    # switch to single strategy selective search method
    gs.switchToSingleStrategy()
  elif (strategy == 'f'):
    # switch to fast but low recall selective search method
    gs.switchToSelectiveSearchFast()
  elif (strategy == 'q'):
    # switch to high recall but slow selective search method
    gs.switchToSelectiveSearchQuality()
  else:
    # otherwise please check the document and exit
    print(__doc__)
    sys.exit(1)

def get_rects(gs):
  # process => calculate all rectangle(region proposal) with strategy
  rects = gs.process()
  # maybe, rects have x,y,w,h (location data). In order to use x:x+w and y:y+h, we use below codes
  rects[:,2] += rects[:,0]
  rects[:,3] += rects[:,1]
  return rects

# if these codes are directly used as interpreter, __name__ has __main__
# if these codes are used as import, __name__ has executeThisModule
if __name__ == '__main__':
  gs = get_selective_search()

  img = cv2.imread('./data/else.jpg',cv2.IMREAD_COLOR)
  config(gs, img, strategy = 'q')

  rects = get_rects(gs)
  print(rects)
