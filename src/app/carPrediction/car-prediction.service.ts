import { Injectable } from '@angular/core';
import * as tf from '@tensorflow/tfjs';

declare var tfvis;

@Injectable({
  providedIn: 'root'
})
export class CarPredictionService {

    constructor() { }


    private async getData (){
        const carsDataReq = await fetch('https://storage.googleapis.com/tfjs-tutorials/carsData.json');  
        const carsData = await carsDataReq.json();  
        const cleaned = carsData.map(car => ({
            mpg: car.Miles_per_Gallon,
            horsepower: car.Horsepower,
        }))
        .filter(car => (car.mpg != null && car.horsepower != null));
        
        return cleaned;
    }

    private async initData ():Promise<any> {
        const data = await this.getData ();
        const values = data.map(d => ({
            x: d.horsepower,
            y: d.mpg,
        }));
        console.log(data);
        tfvis.render.scatterplot(
            {name: 'Horsepower v MPG'},
            {values}, 
            {
                xLabel: 'Horsepower',
                yLabel: 'MPG',
                height: 300
            }
        );
        return data;
    }

    private convertToTensor(data) {
        // Wrapping these calculations in a tidy will dispose any 
        // intermediate tensors.
        
        return tf.tidy(() => {
            // Step 1. Shuffle the data    
            tf.util.shuffle(data);

            // Step 2. Convert data to Tensor
            const inputs = data.map(d => d.horsepower)
            const labels = data.map(d => d.mpg);

            const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);
            const labelTensor = tf.tensor2d(labels, [labels.length, 1]);

            //Step 3. Normalize the data to the range 0 - 1 using min-max scaling
            const inputMax = inputTensor.max();
            const inputMin = inputTensor.min();  
            const labelMax = labelTensor.max();
            const labelMin = labelTensor.min();

            const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));
            const normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin));

            return {
                inputs: normalizedInputs,
                labels: normalizedLabels,
                // Return the min/max bounds so we can use them later.
                inputMax,
                inputMin,
                labelMax,
                labelMin,
            }
        });  
    }

    private async  trainModel(model, inputs, labels) {
        // Prepare the model for training.  
        model.compile({
            optimizer: tf.train.adam(),
            loss: tf.losses.meanSquaredError,
            metrics: ['mse'],
        });
        
        const batchSize = 56;
        const epochs = 50;
        
        return await model.fit(inputs, labels, {
            batchSize,
            epochs,
            shuffle: true,
            callbacks: tfvis.show.fitCallbacks(
                { name: 'Training Performance' },
                ['loss', 'mse'], 
                { height: 200, callbacks: ['onEpochEnd'] }
            )
        });
    }


    public async run (){
        const data = await this.initData ();
        const model = this.createModel();  
        tfvis.show.modelSummary({name: 'Model Summary'}, model);    

        // Convert the data to a form we can use for training.
        const tensorData = this.convertToTensor(data);
        const {inputs, labels} = tensorData;
            
        // Train the model
        console.log('start Training');  
        await this.trainModel(model, inputs, labels);
        console.log('Done Training');

        this.testModel(model, data, tensorData);
    }

    private createModel() {
        // Create a sequential model
        const model = tf.sequential(); 
        
        // Add a single hidden layer
        model.add(tf.layers.dense({inputShape: [1], units: 20, activation: 'sigmoid'}));
        model.add(tf.layers.dense({units: 50, activation: 'sigmoid'}));
        
        // Add an output layer
        model.add(tf.layers.dense({units: 1}));

        return model;
    }

    private testModel(model, inputData, normalizationData) {
        const {inputMax, inputMin, labelMin, labelMax} = normalizationData;  
        
        // Generate predictions for a uniform range of numbers between 0 and 1;
        // We un-normalize the data by doing the inverse of the min-max scaling 
        // that we did earlier.
        const [xs, preds] = tf.tidy(() => {
            
            const xs = tf.linspace(0, 1, 100);      
            const preds = model.predict(xs.reshape([100, 1]));      
            
            const unNormXs = xs
                .mul(inputMax.sub(inputMin))
                .add(inputMin);
            
            const unNormPreds = preds
                .mul(labelMax.sub(labelMin))
                .add(labelMin);
            
            // Un-normalize the data
            return [unNormXs.dataSync(), unNormPreds.dataSync()];
        });
        
        
        const predictedPoints = Array.from(xs).map((val, i) => {
            return {x: val, y: preds[i]}
        });
        
        const originalPoints = inputData.map(d => ({
            x: d.horsepower, y: d.mpg,
        }));
        
        
        tfvis.render.scatterplot(
            {name: 'Model Predictions vs Original Data'}, 
            {values: [originalPoints, predictedPoints], series: ['original', 'predicted']}, 
            {
                xLabel: 'Horsepower',
                yLabel: 'MPG',
                height: 300
            }
        );
    }

    
}
