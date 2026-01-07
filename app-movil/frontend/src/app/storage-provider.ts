// src/app/storage-provider.ts
import { provideZoneChangeDetection, importProvidersFrom } from '@angular/core';
import { IonicStorageModule } from '@ionic/storage-angular';

export const provideAppStorage = () => [
  importProvidersFrom(IonicStorageModule.forRoot())
];
// import { PLATFORM_ID, inject } from '@angular/core';
// import { provideStorage } from '@ionic/storage-angular';
// import { Drivers } from '@ionic/storage';

// export function getStorageProvider() {
//   return provideStorage(
//     inject(PLATFORM_ID),
//     {
//       name: 'tbAppStorage',
//       driverOrder: [Drivers.IndexedDB, Drivers.LocalStorage]
//     }
//   );
// }
