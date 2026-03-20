import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterModule, RouterOutlet } from '@angular/router';

interface NavItem {
  path: string;
  label: string;
  icon: string;
}

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, RouterModule, RouterOutlet],
  template: `
    <!-- Mobile header -->
    <header class="lg:hidden fixed top-0 left-0 right-0 z-30 h-14 flex items-center justify-between px-4
                    bg-surface-900/90 backdrop-blur-lg border-b border-white/[0.06]">
      <div class="flex items-center gap-3">
        <button (click)="sidebarOpen = !sidebarOpen"
                class="w-9 h-9 flex items-center justify-center rounded-lg hover:bg-white/[0.06] transition">
          <svg class="w-5 h-5 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
            <path *ngIf="!sidebarOpen" stroke-linecap="round" d="M4 6h16M4 12h16M4 18h16"/>
            <path *ngIf="sidebarOpen" stroke-linecap="round" d="M6 18L18 6M6 6l12 12"/>
          </svg>
        </button>
        <span class="text-base font-semibold text-white tracking-tight">Options Agent</span>
      </div>
      <div class="flex items-center gap-2">
        <span class="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
        <span class="text-xs text-gray-500">Live</span>
      </div>
    </header>

    <!-- Mobile sidebar backdrop -->
    <div *ngIf="sidebarOpen" class="sidebar-backdrop lg:hidden" (click)="sidebarOpen = false"></div>

    <!-- Sidebar -->
    <aside [class]="'fixed top-0 left-0 bottom-0 z-50 w-[260px] flex flex-col border-r border-white/[0.06] ' +
           'bg-surface-900/95 backdrop-blur-xl transition-transform duration-300 lg:translate-x-0 ' +
           (sidebarOpen ? 'translate-x-0' : '-translate-x-full')">

      <!-- Logo area -->
      <div class="h-16 flex items-center gap-3 px-5 border-b border-white/[0.04]">
        <div class="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center
                    shadow-lg shadow-blue-500/20">
          <svg class="w-4 h-4 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2.5">
            <path stroke-linecap="round" stroke-linejoin="round" d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6"/>
          </svg>
        </div>
        <div>
          <h1 class="text-sm font-bold text-white tracking-tight">Options Agent</h1>
          <p class="text-[10px] text-gray-500 font-medium">AI Trading Dashboard</p>
        </div>
      </div>

      <!-- Nav -->
      <nav class="flex-1 px-3 py-4 space-y-0.5 overflow-y-auto">
        <p class="px-3.5 py-2 text-[10px] font-bold uppercase tracking-[0.15em] text-gray-600">Analytics</p>
        <a *ngFor="let item of navItems; let i = index"
           [routerLink]="item.path"
           routerLinkActive="active"
           class="nav-link"
           (click)="sidebarOpen = false">
          <span class="text-base" [innerHTML]="item.icon"></span>
          <span>{{ item.label }}</span>
        </a>
      </nav>

      <!-- Footer -->
      <div class="px-5 py-4 border-t border-white/[0.04]">
        <div class="flex items-center gap-2.5">
          <span class="w-2 h-2 rounded-full bg-emerald-500"></span>
          <span class="text-xs text-gray-500">Server connected</span>
        </div>
      </div>
    </aside>

    <!-- Main content -->
    <main class="lg:ml-[260px] pt-14 lg:pt-0 min-h-screen">
      <div class="p-4 sm:p-6 lg:p-8 max-w-[1400px] mx-auto">
        <router-outlet></router-outlet>
      </div>
    </main>
  `,
})
export class AppComponent {
  sidebarOpen = false;

  navItems: NavItem[] = [
    { path: '/overview', label: 'Overview', icon: '&#x1F4CA;' },
    { path: '/signals', label: 'Signal Feed', icon: '&#x1F4E1;' },
    { path: '/performance', label: 'Performance', icon: '&#x1F3AF;' },
    { path: '/trade-impact', label: 'Trade Impact', icon: '&#x26A1;' },
    { path: '/tasks', label: 'Tasks', icon: '&#x1F680;' },
  ];
}
