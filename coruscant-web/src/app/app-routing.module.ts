import { NgModule } from '@angular/core';
import { Routes, RouterModule } from '@angular/router';
import {PredictionComponent} from "./prediction/prediction.component";

const routes: Routes = [
  {
    path: 'prediction',
    component: PredictionComponent,
    data: {title: 'Search'},
  },
  {path: '', redirectTo: '/prediction', pathMatch: 'full'},
  {path: '**', component: PredictionComponent},
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
