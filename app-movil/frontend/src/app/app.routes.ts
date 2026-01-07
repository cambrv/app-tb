import { Routes } from '@angular/router';

// @NgModule({
//   ...
//   providers: [{ provide: RouteReuseStrategy, useClass: IonicRouteStrategy
//    }],
// })
export const routes: Routes = [
  {
    path: '',
    redirectTo: 'home',
    pathMatch: 'full',
  },
  {
    path: 'home',
    loadComponent: () => import('./home/home.page').then( m => m.HomePage)
  },
  {
    path: 'chat-recommendation',
    loadComponent: () => import('./chat-recommendation/chat-recommendation.page').then( m => m.ChatRecommendationPage)
  },
];
