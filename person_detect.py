
import numpy as np
import time
from openvino.inference_engine import IECore
import os
import cv2
import argparse
import sys


class Queue:
    '''
    Class for dealing with queues
    '''
    def __init__(self):
        self.queues=[]

    def add_queue(self, points):
        self.queues.append(points)

    def get_queues(self, image):
        for q in self.queues:
            x_min, y_min, x_max, y_max=q
            frame=image[y_min:y_max, x_min:x_max]
            yield frame
    
    def check_coords(self, coords):
        d={k+1:0 for k in range(len(self.queues))}
        for coord in coords:
            for i, q in enumerate(self.queues):
                if coord[0]>q[0] and coord[2]<q[2]:
                    d[i+1]+=1
        return d


class PersonDetect:
    '''
    Class for the Person Detection Model.
    '''

    def __init__(self, model_name, device, threshold=0.60, color='GREEN'):
        self.model_weights=model_name+'.bin'
        self.model_structure=model_name+'.xml'
        self.device=device
        self.threshold=threshold
        self.box_color = color        
        self.network = None

        try:
            self.core = IECore()
            self.model= self.core.read_network(model=self.model_structure, weights=self.model_weights)
        except Exception as e:
            raise ValueError("Could not Initialise the network. Have you enterred the correct model path?")

        self.input_name=next(iter(self.model.inputs))
        self.input_shape=self.model.inputs[self.input_name].shape
        self.output_name=next(iter(self.model.outputs))
        self.output_shape=self.model.outputs[self.output_name].shape

    def load_model(self):
        '''
        TODO: This method needs to be completed by you
        '''
        self.network = self.core.load_network(network=self.model, device_name=self.device, num_requests=1)
        
    def predict(self, frame):
        '''
        TODO: This method needs to be completed by you
        '''
        pre_frame = self.preprocess_input(frame)
        outputs = self.network.infer({self.input_name: pre_frame})
        coords = self.preprocess_outputs(outputs[self.output_name])

        return self.draw_outputs(coords, frame)
    
    def draw_outputs(self, detections, frame):
        '''
        TODO: This method needs to be completed by you
        '''
        width = frame.shape[1]
        height = frame.shape[0]
        boxes = []
        for coords in detections: # Output shape is 1x1xNx7
                xmin = int(coords[0] * width)
                ymin = int(coords[1] * height)
                xmax = int(coords[2] * width)
                ymax = int(coords[3] * height)
                boxes.append([xmin, ymin, xmax, ymax])
                frame = cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), self.convert_color(), 1)
    
        return boxes, frame

    def convert_color(self):
        '''
        Get the BGR value of the desired bounding box color.
        Defaults to Blue if an invalid color is given.
        '''
        colors = {"BLUE": (255,0,0), "GREEN": (0,255,0), "RED": (0,0,255)}
        out_color = colors.get(self.box_color)
        if self.box_color:
            return out_color
        else:
            return colors['GREEN']          
        
    def preprocess_outputs(self, result, width=1, height=1): 
        '''
        TODO: This method needs to be completed by you
        '''

        detections = []
        for box in result[0][0]: # Output shape is 1x1xNx7
            conf = box[2]
            if conf >= self.threshold:
                detections.append([box[3], box[4], box[5], box[6]])
    
        return detections

    def preprocess_input(self, frame):
        '''
        TODO: This method needs to be completed by you
        '''
        pre_frame = cv2.resize(frame, (self.input_shape[3], self.input_shape[2]))
        pre_frame = pre_frame.transpose((2,0,1))
        pre_frame = pre_frame.reshape(1, *pre_frame.shape)      
    
        return pre_frame

def main(args):
    model=args.model
    device=args.device
    video_file=args.video
    max_people=args.max_people
    threshold=args.threshold
    output_path=args.output_path

    start_model_load_time=time.time()
    pd= PersonDetect(model, device, threshold)
    pd.load_model()
    total_model_load_time = time.time() - start_model_load_time

    queue=Queue()
    
    try:
        queue_param=np.load(args.queue_param)
        for q in queue_param:
            queue.add_queue(q)
    except:
        print("error loading queue param file")

    try:
        cap=cv2.VideoCapture(video_file)
    except FileNotFoundError:
        print("Cannot locate video file: "+ video_file)
    except Exception as e:
        print("Something else went wrong with the video file: ", e)
    
    initial_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    initial_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out_video = cv2.VideoWriter(os.path.join(output_path, 'output_video.mp4'), cv2.VideoWriter_fourcc(*'avc1'), fps, (initial_w, initial_h), True)
    
    counter=0
    start_inference_time=time.time()

    try:
        while cap.isOpened():
            ret, frame=cap.read()
            if not ret:
                break
            counter+=1
            
            coords, image= pd.predict(frame)
            num_people= queue.check_coords(coords)
            print(f"Total People in frame = {len(coords)}")
            print(f"Number of people in queue = {num_people}")
            out_text=""
            y_pixel=25
            
            for k, v in num_people.items():
                out_text += f"No. of People in Queue {k} is {v} "
                if v >= int(max_people):
                    out_text += f" Queue full; Please move to next Queue "
                cv2.putText(image, out_text, (15, y_pixel), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                out_text=""
                y_pixel+=40
            out_video.write(image)
            
        total_time=time.time()-start_inference_time
        total_inference_time=round(total_time, 1)
        fps=counter/total_inference_time

        with open(os.path.join(output_path, 'stats.txt'), 'w') as f:
            f.write(str(total_inference_time)+'\n')
            f.write(str(fps)+'\n')
            f.write(str(total_model_load_time)+'\n')

        cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print("Could not run Inference: ", e)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--device', default='CPU')
    parser.add_argument('--video', default=None)
    parser.add_argument('--queue_param', default=None)
    parser.add_argument('--output_path', default='/results')
    parser.add_argument('--max_people', default=2)
    parser.add_argument('--threshold', default=0.60)
    
    args=parser.parse_args()

    main(args)