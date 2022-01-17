import cv2
import os

train_vidnames = ["nightfootage2", "20190128_065933", "footage2", "075812", "155250", "160440"]

vid_path = os.path.join(os.path.dirname(os.getcwd()), "Videos")
annot_path = "Annotations"
images_folder = os.path.join("Training", "Videos")
annotations_folder = os.path.join("Training", "Boxes")

img_no = 1
ann_no = 1

for filename in train_vidnames:
    print(filename)
    vid_filename = os.path.join(vid_path, (filename + ".mp4"))
    vid = cv2.VideoCapture(vid_filename)
    ret = True

    while(ret):
        ret, frame = vid.read()
        if not ret:
            break
        
        img_name = "img" + str(img_no) + ".jpg"
          
        output_name = os.path.join(images_folder, img_name)
            
        cv2.imwrite(output_name, frame)
        
        img_no += 1
        
    anno_filename = os.path.join(annot_path, filename)
    
    onlyfiles = [os.path.join(anno_filename, f) for f in os.listdir(anno_filename) if os.path.isfile(os.path.join(anno_filename, f))]
    
    for file in onlyfiles:
        with open(file, 'r') as x:
            content = x.read()
        
        img_name = "img" + str(ann_no) + ".txt"
        
        output_name = os.path.join(annotations_folder, img_name)
        
        with open(output_name, 'w') as y:
            y.write(content)
            
        ann_no += 1
    
    
    print("Image Number:",img_no)
    print("Annotation Number:", ann_no)
    
    if img_no != ann_no:
        print("Problem")
        break
    
    print("...done")
