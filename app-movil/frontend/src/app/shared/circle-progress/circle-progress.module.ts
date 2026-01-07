import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { NgCircleProgressModule } from 'ng-circle-progress';

@NgModule({
  imports: [
    CommonModule,
    NgCircleProgressModule.forRoot({
      radius: 40,
      outerStrokeWidth: 6,
      innerStrokeWidth: 2,
      animation: true,
      animationDuration: 1000,
      showUnits: true,
      showSubtitle: false,
      showBackground: false,
      startFromZero: true,
      toFixed: 2,
    }),
  ],
  exports: [NgCircleProgressModule],
})
export class CircleProgressModule {}
