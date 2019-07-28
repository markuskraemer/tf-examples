import { HandwrittenRecognitionService } from './../handwritten-recognition.service';
import { classNames } from './../handwritten-recordnition.model.ts';
import { Component, OnInit, Input, ViewChild, ElementRef } from '@angular/core';
import * as tf from '@tensorflow/tfjs';

@Component({
  selector: 'digit-prediction',
  templateUrl: './digit-prediction.component.html',
  styleUrls: ['./digit-prediction.component.scss']
})
export class DigitPredictionComponent implements OnInit {

    private _pictureTensor:any;
    private _labelTensor:any;
    private _predictionTensor:any;

    public label:string = '';

    public predictions:{label:string, percent:number}[] = [];

    @ViewChild('canvas')
    public canvas:ElementRef;

    @Input()
    public set pictureTensor (value:tf.Tensor){
        this._pictureTensor = value;
        this.draw ();
        
    }

    public get pictureTensor(){
        return this._pictureTensor;
    }

    @Input()
    public set labelTensor(value:tf.Tensor2D){
        const hottest = value.argMax(-1);
        this.label = classNames[hottest.dataSync()[0]];
    }

    public get labelTensor(){
        return this._labelTensor;
    }

    @Input()
    public set predictionTensor(value:tf.Tensor2D){
        if(value){
            this._predictionTensor = value;
            this.predictions = this._predictionTensor.dataSync();
        }
    }

    public get predictionTensor (){
        return this._predictionTensor;
    }

    constructor(
        private handwrittenRecognitionService:HandwrittenRecognitionService
    ) { }

    ngOnInit() {
    }

    public predict (): void {
        const tensor = tf.tensor2d(this.pictureTensor.dataSync(), [1, 784]);
        this.handwrittenRecognitionService.predictPictureTensor(<any>tensor);
    }

    private async draw() {
        await tf.browser.toPixels(this._pictureTensor, this.canvas.nativeElement);
    }

}
