import { Component, OnInit, ViewChild, ElementRef } from '@angular/core';
import * as tf from '@tensorflow/tfjs';

class Point {
    public x:number;
    public y:number;
}

@Component({
  selector: 'draw-image',
  templateUrl: './draw-image.component.html',
  styleUrls: ['./draw-image.component.scss']
})
export class DrawImageComponent implements OnInit {

    @ViewChild('canvas')
    public canvas:ElementRef;
    
    public canvasWidth = 300;
    public canvasHeight = 300;

    private isDragging = false;
    private currentMousePoint:Point = {x:0, y:0};
    private context:CanvasRenderingContext2D;

    private datasetImages:Float32Array;

    constructor() { }

    ngOnInit() {
        this.context = this.canvas.nativeElement.getContext('2d');
        this.context.strokeStyle = '#000000';
        this.context.lineWidth = 5;
    }

    public getImageData(){
        return this.context.getImageData(0, 0, this.canvasWidth, this.canvasHeight);
    }

    public predict(){
        this.createDatesetImages ();
      //  this.nextBatch (2, this.datasetImages, ()=>return 0)
    }

    private createDatesetImages():void {
        const NUM_DATASET_ELEMENTS = 1;
        const IMAGE_SIZE = 28*28;

        const datasetBytesBuffer = new ArrayBuffer(NUM_DATASET_ELEMENTS * IMAGE_SIZE * 4);
        const datasetBytesView = new Float32Array(datasetBytesBuffer, 0, IMAGE_SIZE);

        const imageData = this.context.getImageData(0, 0, this.canvasWidth, this.canvasHeight);

        // TODO: mapp imageData to 28 * 28 

        for (let j = 0; j < imageData.data.length / 4; j++) {
            // All channels hold an equal value since the image is grayscale, so
            // just read the red channel.
            datasetBytesView[j] = imageData.data[j * 4] / 255;
        }
    
        this.datasetImages = new Float32Array(datasetBytesBuffer);
    }

    private nextBatch(batchSize, data, index) {
        const IMAGE_SIZE = 28 * 28;
        const batchImagesArray = new Float32Array(batchSize * IMAGE_SIZE);

        for (let i = 0; i < batchSize; i++) {
            const idx = index();

            const image = data[0].slice(idx * IMAGE_SIZE, idx * IMAGE_SIZE + IMAGE_SIZE);
            batchImagesArray.set(image, i * IMAGE_SIZE);

        }

        console.log('nextBatch: ', batchImagesArray);

        const xs = tf.tensor2d(batchImagesArray, [batchSize, IMAGE_SIZE]);

        return xs;
  }



    public handleMouseDown(event:MouseEvent): void{
        this.isDragging = true;
        console.log('md');
        this.context.beginPath();


        this.context.rect(9, 9, 100, 100);

        this.currentMousePoint.x = event.offsetX;
        this.currentMousePoint.y = event.offsetY;
    }

    public handleMouseMove(event:MouseEvent): void{
        if(this.isDragging){
            const newMousePoint = { x: event.offsetX, y:event.offsetY };
            this.draw(this.currentMousePoint, newMousePoint);
            this.currentMousePoint = newMousePoint;
        }
    }

    public handleMouseUp(event:MouseEvent): void {
        this.isDragging = false;
    }

    private draw (from:Point, to:Point){
        console.log('draw: ', from, to);
        this.context.moveTo(from.x, from.y);
        this.context.lineTo(to.x, to.y);
        this.context.stroke();
    }




}
