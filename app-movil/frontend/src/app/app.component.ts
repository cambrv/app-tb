import { Component } from '@angular/core';
import { RouterModule } from '@angular/router';
import { IonRouterOutlet, IonApp } from '@ionic/angular/standalone';
import { Storage } from "@ionic/storage-angular"
@Component({
  selector: 'app-root',
  templateUrl: 'app.component.html',
  styleUrls: ['app.component.scss'],
  standalone: true,
  imports: [
  RouterModule,
  IonRouterOutlet,
  IonApp,
],

})
export class AppComponent {

  constructor(private storage: Storage) {
  this.initializeApp()
  }

  async initializeApp() {
    await this.storage.create()
  }
}
