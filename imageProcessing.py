import os, cv2, numpy as np, copy, sys

filePath="./start"
destinationPath= "./destination"
cascPath = "haarcascade_frontalface_default.xml" 
faceCascade = cv2.CascadeClassifier(cascPath)
print("Start") 
for root, dirs, files in os.walk(filePath, topdown=False): 
    i=10  
    for name in files: 
        frame = cv2.imread(os.path.join(root, name)) 
        try:
          gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
          faces = faceCascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30, 30),flags = cv2.CASCADE_SCALE_IMAGE  )
        except cv2.error:
          print("size=0 error:", sys.exc_info()[0])
        else:
          i=i+1; j=0;
          for(x, y, w, h) in faces:
                if name.split('.', 1 )[1]=="jpeg" :
                     FilenameEX = "jpg"
                else :
                    FilenameEX = name.split('.', 1 )[1]
                parameter = np.array([[0],[0.35],[0.7]]) * np.array([h,w])
                for k, e in enumerate(parameter):
                    image = frame[int(y-e[0]): int(y +h+e[0]), int(x-e[1]): int(x+w+e[1])]
                    fname = str(i) + str(j) +"."+ FilenameEX
                    path=os.path.join(destinationPath,fname)
                    try:
                      cv2.imwrite(path, image)
                    except:
                      print("副檔名錯誤") 
                    j=j+1
              
for root, dirs, files in os.walk(destinationPath, topdown=False): 
    for name in files: 
        size=os.path.getsize(os.path.join(root, name))
        if size < 3*1024 :
           os.remove(os.path.join(root, name)) 

print("Finish") 