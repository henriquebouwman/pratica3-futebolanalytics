import sys,time,cv2,csv,os
from ultralytics import YOLO
video,model,imgsz,a,b=sys.argv[1],sys.argv[2],int(sys.argv[3]),int(sys.argv[4]),int(sys.argv[5])
step=2
out=f"res_{model[:-3]}_{imgsz}.csv"
m=YOLO(model); cap=cv2.VideoCapture(video); cap.set(cv2.CAP_PROP_POS_FRAMES,a)
import pandas as pd
if os.path.exists(out):
  done=pd.read_csv(out).frame.max()+1; a=max(a,int(done)); cap.set(cv2.CAP_PROP_POS_FRAMES,a)
new=not os.path.exists(out); f=open(out,'a'); w=csv.writer(f)
if new: w.writerow(['frame','n_pessoas','n_bola','max_conf_bola','t_inf'])
for i in range(a,b):
  ok,fr=cap.read()
  if not ok: break
  if i%step: continue
  t=time.time(); r=m.predict(fr,imgsz=imgsz,conf=0.10,classes=[0,32],verbose=False,device='cpu')[0]; dt=time.time()-t
  c=r.boxes.cls.numpy(); s=r.boxes.conf.numpy()
  w.writerow([i,int(((c==0)&(s>=0.30)).sum()),int((c==32).sum()),float(s[c==32].max()) if (c==32).any() else 0,round(dt,3)]); f.flush()
