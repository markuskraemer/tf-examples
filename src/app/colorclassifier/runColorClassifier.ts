import * as tf from '@tensorflow/tfjs';
// import * as tfvis from '@tensorflow/tfjs-vis';

declare var tfvis;

let labelCategorys:string[];

async function getData () {
    const f = await fetch('assets/colorData.json');
    return f.json ();
}

function createModel (outputCount:number):tf.Sequential {

    const model:tf.Sequential = tf.sequential ();

    const hidden = tf.layers.dense({inputShape: [3], units: 4, activation:'softmax'});
    const output = tf.layers.dense({units: outputCount, activation:'softmax'});

    model.add(hidden);
    model.add(output);

    model.compile ({
        optimizer:tf.train.sgd(.2),
        loss:'categoricalCrossentropy'
    })

    return model;

}

function normalize (data:IColorData[]):IColorData[]{
    return data.map ((i) => {
        return {
            r:i.r / 255,
            g:i.g / 255,
            b:i.b / 255,
            label:i.label
        }
    })
}

function initLabels (data:IColorData[]) {
    labelCategorys = [];
    data.forEach (i => {
        if(!labelCategorys.includes (i.label)){
            labelCategorys.push(i.label);
        }
    })
}

function convertToTensors (data:IColorData[]):{xs:tf.Tensor, ys:tf.Tensor}{
    const inputs = data.map(d => [d.r, d.g, d.b]);
    const labels = data.map(d => labelCategorys.indexOf(d.label));
    const ys = tf.oneHot (labels, labelCategorys.length);

    return { xs: tf.tensor2d(inputs, [inputs.length, 3], 'int32'),
             ys: ys };
}

async function trainModel(model:tf.Sequential, xs:tf.Tensor, ys:tf.Tensor) {
  
  const batchSize = 50;
  const epochs = 100;

  return await model.fit(xs, ys, {
    batchSize,
    epochs,
    shuffle: true,
    callbacks:  tfvis.show.fitCallbacks ( 
                { name: 'Training Performance' },
                ['loss', 'mse'], 
                { height: 200, callbacks: ['onEpochEnd'] })    
  });
}

async function run () {
    const data:IColorData[] = await getData ();       
    data.length = 11; 
    const normalizedData = normalize (data);
    initLabels (data);

    const model = createModel (labelCategorys.length);
    const {xs, ys} = convertToTensors(data);

    ys.print ();

    console.log('startTrain');
    trainModel (model, xs, ys);
    console.log('endTrain');

}

export function runColorClassifier ():void {
    console.log('runColorClassifier');
    run ();
}