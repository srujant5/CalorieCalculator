#!/usr/bin/python3
import jetson_inference
import jetson_utils
import argparse
import csv
import sys
import inspect
# create video sources & outputs
parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str, help="filename of the image to process")
parser.add_argument("output", type=str, default="output.jpg", nargs='?', help="URI of the output stream")
parser.add_argument("--topK", type=int, default=1, help="show the topK number of class predictions (default: 1)")

opt = parser.parse_args()

input = jetson_utils.videoSource(opt.filename, argv=sys.argv)
output = jetson_utils.videoOutput(opt.output, argv=sys.argv)
font = jetson_utils.cudaFont()
net = jetson_inference.imageNet(model="resnet18.onnx",labels="labels.txt",input_blob="input_0",output_blob="output_0")
ftc = {}
with open("foodtocal.csv", "r") as f:
    for line in f:
        pair=line.strip("\n").split(",")
        ftc[pair[0]]=pair[1]
label=""
final_confidence=""
while True:
    # capture the next image
    img = input.Capture()
    if img is None: # timeout
        continue  

    # classify the image and get the topK predictions
    # if you only want the top class, you can simply run:
    #   class_id, confidence = net.Classify(img)
    predictions = net.Classify(img, topK=opt.topK)
    
    # draw predicted class labels
    for n, (classID, confidence) in enumerate(predictions):
        classLabel = net.GetClassLabel(classID)
        confidence *= 100.0
        label=str(classLabel)
        final_confidence=confidence
        font.OverlayText(img,text=ftc[classLabel], 
                         x=5, y=5 + n * (font.GetSize() + 5),
                         color=font.White, background=font.Gray40)
        font.OverlayText(img, text=f"{classLabel} ({confidence:05.2f}%)", 
                         x=5, y=img.height-((font.GetSize() + 5)),
                         color=font.White, background=font.Gray40)
                         
    # render the image
    output.Render(img)

    # update the title bar
    output.SetStatus("{:s} | Network {:.0f} FPS".format(net.GetNetworkName(), net.GetNetworkFPS()))

    # print out performance info
    net.PrintProfilerTimes()

    # exit on input/output EOS
    if not input.IsStreaming() or not output.IsStreaming():
        break


print("image is recognized as "+ label +" with " + str(final_confidence)+"% confidence("+ftc[label]+")")
