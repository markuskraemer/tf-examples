import { CarPredictionService } from './../car-prediction.service';
import { Component, OnInit } from '@angular/core';

@Component({
  selector: 'car-prediction',
  templateUrl: './car-prediction.component.html',
  styleUrls: ['./car-prediction.component.css']
})
export class CarPredictionComponent implements OnInit {

  constructor(
      public carPredictionService:CarPredictionService
  ) { }

  ngOnInit() {
      this.carPredictionService.run ();
  }

}
