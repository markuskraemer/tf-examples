import { ReplaySubject } from 'rxjs';
import { ColorClassifierService } from './../color-classifier.service';
import { Component, OnInit, Input, ChangeDetectionStrategy } from '@angular/core';

@Component({
  selector: 'color-label',
  templateUrl: './color-label.component.html',
  styleUrls: ['./color-label.component.scss'],
  changeDetection:ChangeDetectionStrategy.OnPush
})
export class ColorLabelComponent implements OnInit {

    private _color:Color = {r:0, g:0, b:0};
    public prediction$ = new ReplaySubject<string> ();

    @Input ()
    public set color (value:Color){
        this._color = value;
        this.updateLabel ();
    }

    public get color ():Color {
        return this._color;
    }

    public updateLabel ():void {
        this.colorClassifierService.getLabel (this._color).subscribe (
            value => {
              //  console.log('new label: ', value);
                this.prediction$.next(value);
            }
        );
    }

    constructor(
        public readonly colorClassifierService:ColorClassifierService
    ) { }

    ngOnInit() {
    }

}
