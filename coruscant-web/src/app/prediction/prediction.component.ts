import { Component, OnInit } from '@angular/core';
import {MatSnackBar} from '@angular/material/snack-bar';
import {FormBuilder, FormControl, FormGroup, NgForm, Validators} from '@angular/forms';
import {PredictionService} from "../prediction-service/prediction.service";
import {Prediction} from "../model/prediction";

@Component({
  selector: 'app-prediction',
  templateUrl: './prediction.component.html',
  styleUrls: ['./prediction.component.css']
})
export class PredictionComponent implements OnInit {

  form: FormGroup;
  isLoadingResults = true;
  model: Prediction;
  subtitle: string;
  constructor(private formBuilder: FormBuilder, private service: PredictionService, private snackBar: MatSnackBar) { }

  ngOnInit(): void {
    this.isLoadingResults = false;
    this.model = new Prediction();
    this.form = this.formBuilder.group({
      textSearch: [null, Validators.required]
    });
  }

  onFormSubmit() {
    this.isLoadingResults = true;
    this.service.getPredictions(this.form.value).subscribe(
      res => {
        this.model = res;

        this.isLoadingResults = false;
      },
      err => {
        this.isLoadingResults = false;
        this.openSnackBar(err.message, 'Error');
      }
    );
  }

  openSnackBar(message: any, action: string) {
    this.snackBar.open(message, action, {
      duration: 5000,
      verticalPosition: 'top',
      horizontalPosition: 'end',
    });
  }
}
