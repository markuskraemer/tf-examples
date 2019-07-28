import { classNames } from './handwritten-recordnition.model.ts';
import { Injectable } from '@angular/core';
import * as tf from '@tensorflow/tfjs';
import { MnistData } from './data.js';

declare var tfvis;

@Injectable({
  providedIn: 'root'
})
export class HandwrittenRecognitionService {

    private model:tf.Sequential;
    private data:MnistData;

    private IMAGE_WIDTH = 28;
    private IMAGE_HEIGHT = 28;

    constructor() { }

    public getImage (count){
        const batch = this.data.nextTestBatch (count);
        console.log('batch: ', batch);
        return batch;
    }

    public async run() {  
        console.log('run');

        this.data = new MnistData();
        await this.data.load();
        await this.showExamples(this.data);

        this.model = this.getModel();
        tfvis.show.modelSummary({name: 'Model Architecture'}, this.model);
  
        console.log('BACKEND: ', tf.getBackend());

        if(false){

            console.log('--- train ---');
            await this.train(this.model, this.data);
            	
            /*
            console.log('--- showAccurary ---');
            await this.showAccuracy(this.model, data);

            console.log('--- showConfusion ---');
            await this.showConfusion(this.model, data);
            */
        }
    }

    private async train(model, data) {
        const metrics = ['loss', 'acc',];
        const container = {
            name: 'Model Training', styles: { height: '1000px' }
        };
        const fitCallbacks = tfvis.show.fitCallbacks(container, metrics);
        console.log('fitCallbacks: ', fitCallbacks);
        const BATCH_SIZE = 5;
        const TRAIN_DATA_SIZE = 10;
        const TEST_DATA_SIZE = 10;

        const [trainXs, trainYs] = tf.tidy(() => {
            const d = data.nextTrainBatch(TRAIN_DATA_SIZE);
            return [
                d.xs.reshape([TRAIN_DATA_SIZE, this.IMAGE_WIDTH, this.IMAGE_HEIGHT, 1]),
                d.labels
            ];
        });

        const [testXs, testYs] = tf.tidy(() => {
            const d = data.nextTestBatch(TEST_DATA_SIZE);
            return [
                d.xs.reshape([TEST_DATA_SIZE, this.IMAGE_WIDTH, this.IMAGE_HEIGHT, 1]),
                d.labels
            ];
        });

        return model.fit(
            trainXs, trainYs, {
                batchSize: BATCH_SIZE,
                validationData: [testXs, testYs],
                epochs: 10,
                shuffle: true,
                callbacks: {
                    onTrainBegin: (...args) => {
                        console.log('onTrainhBegin ', args);
                    },
                    onTrainEnd: (...args) => {
                        console.log('onTrainhEnd ', args);
                    },
                    onEpochEnd: (...args) => {
                        console.log('onEpochEnd ', args);
                        fitCallbacks.onEpochEnd(args[0], args[1]);
                    },
                    onBatchEnd: (...args) => {
                        console.log('onBatchEnd ', args);
                        fitCallbacks.onBatchEnd (args[0], args[1]);
                    }

                }
        });
    }

    private async showExamples(data) {
        // Create a container in the visor
        const surface = tfvis.visor().surface({ name: 'Input Data Examples', tab: 'Input Data'});  

        // Get the examples
        const examples = data.nextTestBatch(2);
        const numExamples = examples.xs.shape[0];
        
        // Create a canvas element to render each example
        for (let i = 0; i < numExamples; i++) {
            const imageTensor = tf.tidy(() => {
                // Reshape the image to 28x28 px
                return examples.xs
                    .slice([i, 0], [1, examples.xs.shape[1]])
                    .reshape([this.IMAGE_WIDTH, this.IMAGE_HEIGHT, 1]);
            });
            
            const canvas = document.createElement('canvas');
            canvas.width = this.IMAGE_WIDTH, this.IMAGE_HEIGHT;
            canvas.height = this.IMAGE_WIDTH, this.IMAGE_HEIGHT;
            canvas.style.margin = '4px';
            await tf.browser.toPixels(imageTensor, canvas);
            surface.drawArea.appendChild(canvas);

            imageTensor.dispose();
        
        }
    }

    private getModel() {
        const model = tf.sequential();
        
        const IMAGE_CHANNELS = 1;  
        
        // In the first layer of our convolutional neural network we have 
        // to specify the input shape. Then we specify some parameters for 
        // the convolution operation that takes place in this layer.
        model.add(tf.layers.conv2d({
            inputShape: [this.IMAGE_WIDTH, this.IMAGE_HEIGHT, IMAGE_CHANNELS],
            kernelSize: 5,
            filters: 8,
            strides: 1,
            activation: 'relu',
            kernelInitializer: 'varianceScaling'
        }));

        // The MaxPooling layer acts as a sort of downsampling using max values
        // in a region instead of averaging.  
        model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));
        
        // Repeat another conv2d + maxPooling stack. 
        // Note that we have more filters in the convolution.
        model.add(tf.layers.conv2d({
            kernelSize: 5,
            filters: 16,
            strides: 1,
            activation: 'relu',
            kernelInitializer: 'varianceScaling'
        }));
        model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));
        
        // Now we flatten the output from the 2D filters into a 1D vector to prepare
        // it for input into our last layer. This is common practice when feeding
        // higher dimensional data to a final classification output layer.
        model.add(tf.layers.flatten());

        // Our last layer is a dense layer which has 10 output units, one for each
        // output class (i.e. 0, 1, 2, 3, 4, 5, 6, 7, 8, 9).
        const NUM_OUTPUT_CLASSES = 10;
        model.add(tf.layers.dense({
            units: NUM_OUTPUT_CLASSES,
            kernelInitializer: 'varianceScaling',
            activation: 'softmax'
        }));

        
        // Choose an optimizer, loss function and accuracy metric,
        // then compile and return the model
        const optimizer = tf.train.adam();
        model.compile({
            optimizer: optimizer,
            loss: 'categoricalCrossentropy',
            metrics: ['accuracy'],
        });

        return model;
    }

    public predict (data:{xs, labels}){
        const testxs = data.xs.reshape([1, this.IMAGE_WIDTH, this.IMAGE_HEIGHT, 1]);
        const labels = data.labels.argMax([-1]);
        const preds = this.model.predict(testxs)['argMax']([-1]);

        testxs.dispose();
        return [preds, labels];  
    }

    public predict2 (data:any){
        const testxs = data.xs.reshape([data.xs.shape[0], this.IMAGE_WIDTH, this.IMAGE_HEIGHT, 1]);
        const labels = data.labels.argMax([-1]);
        const preds = this.model.predict(testxs);
        
        const predArgMax = preds['argMax']([-1]);

        console.log('labels: ');
        tf.print(labels);
        
        console.log('preds');
        tf.print(preds as any);


        console.log('predArgMax');
        tf.print(predArgMax as any);


        testxs.dispose();
        return [preds, labels];
    }

    public predictPictureTensor (xs:tf.Tensor2D):tf.Tensor | tf.Tensor[]{
        const testxs = xs.reshape([xs.shape[0], this.IMAGE_WIDTH, this.IMAGE_HEIGHT, 1]);
        const preds = this.model.predict(testxs);
        return preds;
    }


    private doPrediction(model, data, testDataSize = 500) {
        const testData = data.nextTestBatch(testDataSize);
        const testxs = testData.xs.reshape([testDataSize, this.IMAGE_WIDTH, this.IMAGE_HEIGHT, 1]);
        const labels = testData.labels.argMax([-1]);
        const preds = model.predict(testxs).argMax([-1]);

        testxs.dispose();
        return [preds, labels];
    }

    private async showAccuracy(model, data) {
        const [preds, labels] = this.doPrediction(model, data);
        const classAccuracy = await tfvis.metrics.perClassAccuracy(labels, preds);
        const container = {name: 'Accuracy', tab: 'Evaluation'};
        tfvis.show.perClassAccuracy(container, classAccuracy, classNames);

        labels.dispose();
    }

    private async showConfusion(model, data) {
        const [preds, labels] = this.doPrediction(model, data);
        const confusionMatrix = await tfvis.metrics.confusionMatrix(labels, preds);
        const container = {name: 'Confusion Matrix', tab: 'Evaluation'};
        tfvis.render.confusionMatrix(
            container, {values: confusionMatrix}, classNames);

        labels.dispose();
    }

}
