import { HandwrittenRecognitionService } from '../handwritten-recognition.service';
import { Component, OnInit, ViewChild, ElementRef } from '@angular/core';
import * as tf from '@tensorflow/tfjs';

@Component({
    selector: 'handwritten-recognition',
    templateUrl: './handwritten-recognition.component.html',
    styleUrls: ['./handwritten-recognition.component.css']
})
export class HandwrittenRecognitionComponent implements OnInit {

    @ViewChild('drawarea')
    public drawArea:ElementRef;

    public data:{xs:tf.Tensor, labels:tf.Tensor};

    public pictures:any[] = [];
    public labels:any[] = [];
    public predictions:any[] = [];

    constructor(
        private handwrittenRecognitionService:HandwrittenRecognitionService
    ) { }

    ngOnInit() {
        this.handwrittenRecognitionService.run ();
        setTimeout(async ()=>{
            this.data = this.handwrittenRecognitionService.getImage (2);
            await this.showPic(this.data);
        }, 7000);
    }

    public async predict() {
        const preds = <any>this.handwrittenRecognitionService.predictPictureTensor(<any>this.data.xs);
        console.log('pred: ', preds, preds['dataSync']());

        const numPreds = preds.shape[0];

        for(let i = 0; i < numPreds; ++i){
            const predTensor = tf.tidy(() => {
                return preds
                    .slice([i, 0], [1, preds.shape[1]]);
            })
            this.predictions.push(predTensor);
        }
    }

    private async showPic(pics) {
        console.log('showPic: ', pics);

        tf.print(pics.xs);
        tf.print(pics.labels);

        const prediction = this.handwrittenRecognitionService.predict2 (pics);
        console.log('prediction: ', prediction);
        tf.print (prediction as any);
        // Get the examples
        const numExamples = pics.xs.shape[0];
        
        // Create a canvas element to render each example
        for (let i = 0; i < numExamples; i++) {
            const imageTensor = tf.tidy(() => {
                // Reshape the image to 28x28 px
                return pics.xs
                    .slice([i, 0], [1, pics.xs.shape[1]])
                    .reshape([28, 28]);
            });
            //console.log('imageTensor: ', imageTensor);
            this.pictures.push (imageTensor);
            
            const labelTensor = tf.tidy(() => {
                return pics.labels
                    .slice([i, 0], [1, pics.labels.shape[1]]);
            })
            this.labels.push(labelTensor);
           
            //console.log('labelTensor: ', labelTensor);


            const canvas = document.createElement('canvas');
            canvas.width = 28;
            canvas.height = 28;
            canvas.style.margin = '4px';
            await tf.browser.toPixels(imageTensor, canvas);
            this.drawArea.nativeElement.appendChild(canvas);

            //imageTensor.dispose();
            // labelTensor.dispose ();
        }
    }
}
