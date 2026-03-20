import { Routes } from '@angular/router';

export const routes: Routes = [
  { path: '', redirectTo: 'overview', pathMatch: 'full' },
  {
    path: 'overview',
    loadComponent: () => import('./pages/overview/overview.component').then(m => m.OverviewComponent),
  },
  {
    path: 'signals',
    loadComponent: () => import('./pages/signals/signals.component').then(m => m.SignalsComponent),
  },
  {
    path: 'performance',
    loadComponent: () => import('./pages/performance/performance.component').then(m => m.PerformanceComponent),
  },
  {
    path: 'trade-impact',
    loadComponent: () => import('./pages/trade-impact/trade-impact.component').then(m => m.TradeImpactComponent),
  },
  {
    path: 'tasks',
    loadComponent: () => import('./pages/tasks/tasks.component').then(m => m.TasksComponent),
  },
  { path: '**', redirectTo: 'overview' },
];
