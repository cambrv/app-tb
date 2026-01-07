import { bootstrapApplication } from '@angular/platform-browser';
import { enableProdMode } from "@angular/core";
import {
  RouteReuseStrategy,
  provideRouter,
  withPreloading,
  PreloadAllModules,
} from '@angular/router';
import { IonicRouteStrategy, provideIonicAngular } from "@ionic/angular/standalone"
import { provideHttpClient } from "@angular/common/http"
import { Storage } from "@ionic/storage-angular"
import { routes } from "./app/app.routes"
import { AppComponent } from "./app/app.component"
import { environment } from "./environments/environment"
import { provideAnimations } from '@angular/platform-browser/animations';

if (environment.production) {
  enableProdMode()
}

bootstrapApplication(AppComponent, {
  providers: [
    { provide: RouteReuseStrategy, useClass: IonicRouteStrategy },
    provideIonicAngular(),
    provideRouter(routes, withPreloading(PreloadAllModules)),
    provideAnimations(),
    provideHttpClient(),
    Storage
  ],
});
