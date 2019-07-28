import { DrawImageComponent } from './handwrittenRecognition/draw-image/draw-image.component';
import { DigitPredictionComponent } from './handwrittenRecognition/digit-prediction/digit-prediction.component';
import { HandwrittenRecognitionService } from './handwrittenRecognition/handwritten-recognition.service';
import { HandwrittenRecognitionComponent } from './handwrittenRecognition/handwritten-recognition/handwritten-recognition.component';

import { CarPredictionService } from './carPrediction/car-prediction.service';
import { CarPredictionComponent } from './carPrediction/car-prediction/car-prediction.component';
import { ColorLabelComponent } from './colorclassifier/color-label/color-label.component';
import { ColorClassifierService } from './colorclassifier/color-classifier.service';
import { ColorClassifierComponent } from './colorclassifier/color-classifier/color-classifier.component';
import { BrowserModule } from '@angular/platform-browser';
import { NgModule } from '@angular/core';

import { AppComponent } from './app.component';

@NgModule({
  declarations: [
    AppComponent,
    ColorClassifierComponent,
    ColorLabelComponent,
    CarPredictionComponent,
    HandwrittenRecognitionComponent,
    DigitPredictionComponent,
    DrawImageComponent
],
  imports: [
    BrowserModule
  ],
  providers: [
      ColorClassifierService,
      CarPredictionService,
      HandwrittenRecognitionService,
      ],
  bootstrap: [AppComponent]
})
export class AppModule { }
