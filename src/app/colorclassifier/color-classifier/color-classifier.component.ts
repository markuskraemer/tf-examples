import { ColorLabelComponent } from './../color-label/color-label.component';

import { BehaviorSubject, Subject, of, Observable, ReplaySubject } from 'rxjs';
import { tap, map } from 'rxjs/operators';
import { ColorClassifierService } from './../color-classifier.service';
import { Component, OnInit, ChangeDetectionStrategy, ViewChildren } from '@angular/core';

@Component({
  selector: 'color-classifier',
  templateUrl: './color-classifier.component.html',
  styleUrls: ['./color-classifier.component.scss'],
  changeDetection:ChangeDetectionStrategy.OnPush 
})
export class ColorClassifierComponent implements OnInit {

    @ViewChildren ('colorLabels')
    public colorLabels:ColorLabelComponent[];

    public colorData$ = new ReplaySubject<IColorData[]>();    
   
    public colors:Color[] = [];

    constructor(
        public colorClassifierService:ColorClassifierService
    ) { 
        this.createColors (10);
    }

    ngOnInit() {
        this.colorClassifierService.run ();

        this.colorClassifierService.data$.subscribe (
           (data) => {
               this.colorData$.next(data.slice(0,9))
           }
       )

       console.log('colorLabels', this.colorLabels);
    }

    private createColors (count:number){
        for(let i = 0; i < count; ++i){
            this.colors.push(this.getColor());
        }
    }

    private getColor ():Color {
        return  {
            r:Math.random () * 255 | 0,
            g:Math.random () * 255 | 0,
            b:Math.random () * 255 | 0,
        };
    }

    public updateColor (index:number):void {
        this.colors[index] = this.getColor ();
    }

    public updateLabels ():void {
        this.colorLabels.forEach (colorLabel => colorLabel.updateLabel ());
    }

}
