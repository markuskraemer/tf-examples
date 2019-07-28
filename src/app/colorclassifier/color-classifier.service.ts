
import { Injectable } from '@angular/core';
import * as tf from '@tensorflow/tfjs';
import { ReplaySubject, BehaviorSubject, Observable, Subject } from 'rxjs';

declare var tfvis;

@Injectable({
  providedIn: 'root'
})
export class ColorClassifierService {

    private model:tf.Sequential
    private labelCategorys:string[];
    public loss: number;
    public readonly data$ = new ReplaySubject<IColorData[]> ();
    constructor() { }

    public async run () {
        const data:IColorData[] = await this.getData ();
            
        const normalizedData = <IColorData[]>this.normalize (data);
        this.initLabels (data);


        this.model = this.createModel (this.labelCategorys.length);
        
        const {xs, ys} = this.convertToTensors(normalizedData);

        this.data$.next(data);

        ys.print ();

        console.log('startTrain');
        this.trainModel (xs, ys);
        console.log('endTrain');
    }

    public getLabel (o:Color) {
        const oAsArray = this.normalize([o])[0];
        const rgb = [oAsArray.r, oAsArray.g, oAsArray.b];
        const xs = tf.tensor2d(rgb, [1, 3], 'float32');
        const subject = new ReplaySubject<string> ();
        console.log('getLabel: ', rgb);

        if(this.model) {
            tf.tidy (() => {    
                const prediction = <any>this.model.predict(xs);
                const idx = prediction.argMax(1).dataSync()[0];
                console.log('  label: ' + idx + ' ' + this.labelCategorys[idx]);
                
                // return idx + ' ' + this.labelCategorys[idx];
                subject.next(idx + ' ' + this.labelCategorys[idx]);
             });
        }
        return subject;
    }

    private async getData () {
        const f = await fetch('assets/colorData.json');
        return f.json ();
    }

    private convertToTensors (data:IColorData[]):{xs:tf.Tensor, ys:tf.Tensor}{
        const inputs = data.map(d => [d.r, d.g, d.b]);
        const labels = data.map(d => this.labelCategorys.indexOf(d.label));
        const ys = tf.oneHot (labels, this.labelCategorys.length);

        return { xs: tf.tensor2d(inputs, [inputs.length, 3], 'float32'),
                ys: ys };
    }

    private normalize (data:Color[]):Color[]{
        return data.map ((i) => {
            let {...copy} = i;
            copy.r /= 255;
            copy.g /= 255;
            copy.b /= 255;
            return copy;
        })
    }

    private initLabels (data:IColorData[]) {
        this.labelCategorys = [];
        data.forEach (i => {
            if(!this.labelCategorys.includes (i.label)){
                this.labelCategorys.push(i.label);
            }
        })
    }


    private createModel (outputCount:number):tf.Sequential {

        const model:tf.Sequential = tf.sequential ();

        const hidden = tf.layers.dense({inputDim: 3, units: 4, activation:'sigmoid'});
        const output = tf.layers.dense({units: outputCount, activation:'softmax'});

        model.add(hidden);
        model.add(output);

        model.compile ({
            optimizer:tf.train.sgd(.2),
            loss:'categoricalCrossentropy'
        })

        return model;
    }

    private async trainModel(xs:tf.Tensor, ys:tf.Tensor) {
    
        const batchSize = 50;
        const epochs = 150;

        return await this.model.fit(xs, ys, {
            epochs:epochs,
            shuffle: true,
            callbacks:             
                tfvis.show.fitCallbacks ( 
                            { name: 'Training Performance' },
                            ['loss'],  
                            { height: 200, callbacks: ['onEpochEnd'] }
                        )    
        });
    }

}




