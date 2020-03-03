import {Injectable} from '@angular/core';
import {environment} from '../../environments/environment';
import {HttpClient} from "@angular/common/http";
import {tap} from 'rxjs/operators';
import {Observable} from "rxjs";
import {Prediction} from "../model/prediction";

@Injectable({
  providedIn: 'root'
})
export class PredictionService {

  private apiUrl: string = environment.urlEndPoint;

  constructor(private http: HttpClient) {
  }

  getPredictions(data: Prediction): Observable<Prediction> {
    return this.http.post<Prediction>(this.apiUrl, data).pipe(
      tap(res => res, error => this.handleError(error)));
  }

  private handleError(error: any) {
    console.log('Error prediction service: %s', error);
    throw error;
  }
}
