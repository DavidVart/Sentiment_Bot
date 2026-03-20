import { Component, OnInit, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ApiService, TaskRun } from '../../services/api.service';
import { Subscription, interval, switchMap, of } from 'rxjs';

@Component({
  selector: 'app-tasks',
  standalone: true,
  imports: [CommonModule, FormsModule],
  template: `
    <div class="space-y-6 animate-fade-in">
      <!-- Header -->
      <div class="flex items-center justify-between flex-wrap gap-4">
        <h1 class="page-title flex items-center gap-3">
          <span>Task Monitor</span>
          <span *ngIf="activeTasks.length > 0"
                class="inline-flex items-center justify-center w-6 h-6 rounded-full bg-blue-500/20 text-blue-400 text-xs font-bold animate-pulse-glow">
            {{ activeTasks.length }}
          </span>
        </h1>
        <button (click)="showLaunchModal = true" class="btn-primary flex items-center gap-2">
          <svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2.5">
            <path stroke-linecap="round" d="M12 4v16m8-8H4"/>
          </svg>
          Launch Task
        </button>
      </div>

      <!-- Error -->
      <div *ngIf="error" class="glass-card border-red-500/20 text-red-400 text-sm flex items-center justify-between">
        <span>{{ error }}</span>
        <button (click)="error = null" class="text-red-300 hover:text-red-200 text-xs">&times;</button>
      </div>

      <!-- Active tasks -->
      <div *ngIf="activeTasks.length > 0" class="space-y-4">
        <h2 class="section-title">Active</h2>
        <div *ngFor="let t of activeTasks" class="glass-card animate-slide-up">
          <!-- Top row -->
          <div class="flex items-start justify-between gap-4 mb-3">
            <div>
              <h3 class="text-white font-medium">{{ t.task_label || t.task_type }}</h3>
              <span class="badge badge-blue mt-1">{{ t.status }}</span>
            </div>
            <div class="flex items-center gap-3 text-sm text-gray-400 shrink-0">
              <span class="tabular-nums flex items-center gap-1.5">
                <svg class="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
                  <circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/>
                </svg>
                {{ fmtDuration(t.elapsed_seconds) }}
              </span>
              <span *ngIf="t.eta_seconds != null" class="tabular-nums text-gray-500">
                ETA {{ fmtDuration(t.eta_seconds) }}
              </span>
              <button *ngIf="t.status === 'running'" (click)="handleCancel(t.id)" class="btn-danger">
                Cancel
              </button>
            </div>
          </div>

          <!-- Progress bar -->
          <div class="progress-bar mb-2">
            <div class="progress-fill" [style.width.%]="Math.min(t.progress_pct ?? 0, 100)"></div>
          </div>
          <div class="flex justify-between text-xs text-gray-400 mb-3">
            <span class="truncate mr-4">{{ t.detail || 'Starting...' }}</span>
            <span class="tabular-nums shrink-0 font-medium">{{ (t.progress_pct ?? 0).toFixed(1) }}%</span>
          </div>

          <!-- Log toggle -->
          <button (click)="toggleExpanded(t.id)" class="text-xs text-blue-400 hover:text-blue-300 transition flex items-center gap-1">
            <svg class="w-3 h-3 transition-transform" [class.rotate-90]="expandedId === t.id" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2.5">
              <path stroke-linecap="round" d="M9 5l7 7-7 7"/>
            </svg>
            {{ expandedId === t.id ? 'Hide logs' : 'Show logs' }}
          </button>
          <pre *ngIf="expandedId === t.id && expandedTask?.log_tail"
               class="mt-3 p-4 rounded-xl bg-black/40 border border-white/[0.04] text-xs text-gray-300 overflow-x-auto max-h-64 overflow-y-auto whitespace-pre-wrap"
               style="font-family: var(--font-mono)">{{ expandedTask!.log_tail }}</pre>
        </div>
      </div>

      <!-- No active tasks -->
      <div *ngIf="activeTasks.length === 0" class="glass-card text-center py-16">
        <div class="w-16 h-16 mx-auto mb-4 rounded-2xl bg-white/[0.04] flex items-center justify-center">
          <svg class="w-8 h-8 text-gray-600" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.5">
            <path stroke-linecap="round" d="M5 3l14 9-14 9V3z"/>
          </svg>
        </div>
        <p class="text-gray-400 font-medium">No active tasks</p>
        <p class="text-sm text-gray-600 mt-1">Click <strong>Launch Task</strong> to start one.</p>
      </div>

      <!-- History -->
      <div *ngIf="historyTasks.length > 0">
        <h2 class="section-title mb-4">History</h2>
        <div class="glass-card overflow-x-auto !p-0">
          <table class="w-full text-sm">
            <thead>
              <tr class="text-left text-gray-500 border-b border-white/[0.06]">
                <th class="p-4 font-medium">Task</th>
                <th class="p-4 font-medium">Status</th>
                <th class="p-4 font-medium hidden sm:table-cell">Duration</th>
                <th class="p-4 font-medium hidden md:table-cell">Started</th>
                <th class="p-4 font-medium">Actions</th>
              </tr>
            </thead>
            <tbody>
              <tr *ngFor="let t of historyTasks"
                  class="border-b border-white/[0.03] hover:bg-white/[0.02] transition">
                <td class="p-4 text-white font-medium">{{ t.task_label || t.task_type }}</td>
                <td class="p-4">
                  <span class="badge" [class]="statusBadge(t.status)">{{ t.status }}</span>
                </td>
                <td class="p-4 text-gray-400 tabular-nums hidden sm:table-cell">{{ fmtDuration(t.elapsed_seconds) }}</td>
                <td class="p-4 text-gray-400 text-xs hidden md:table-cell">
                  {{ t.started_at ? fmtDate(t.started_at) : '\u2014' }}
                </td>
                <td class="p-4">
                  <button (click)="toggleExpanded(t.id)" class="text-xs text-blue-400 hover:text-blue-300 transition">
                    {{ expandedId === t.id ? 'Hide' : 'Details' }}
                  </button>
                </td>
              </tr>
            </tbody>
          </table>
        </div>

        <!-- Expanded detail -->
        <div *ngIf="expandedId && isHistoryExpanded() && expandedTask" class="glass-card mt-3 animate-fade-in">
          <h3 class="text-sm font-medium text-white mb-3">{{ expandedTask.task_label || expandedTask.task_type }}</h3>
          <pre *ngIf="expandedTask.error_message"
               class="mb-3 p-4 rounded-xl bg-red-500/10 border border-red-500/20 text-xs text-red-300 overflow-x-auto whitespace-pre-wrap"
               style="font-family: var(--font-mono)">{{ expandedTask.error_message }}</pre>
          <pre *ngIf="expandedTask.log_tail"
               class="p-4 rounded-xl bg-black/40 border border-white/[0.04] text-xs text-gray-300 overflow-x-auto max-h-64 overflow-y-auto whitespace-pre-wrap"
               style="font-family: var(--font-mono)">{{ expandedTask.log_tail }}</pre>
        </div>
      </div>

      <!-- Launch Modal -->
      <div *ngIf="showLaunchModal" class="fixed inset-0 z-[60] flex items-end sm:items-center justify-center p-4"
           (click)="showLaunchModal = false">
        <div class="fixed inset-0 bg-black/70 backdrop-blur-sm"></div>
        <div class="relative w-full max-w-md rounded-2xl border border-white/[0.08] p-6 animate-slide-up"
             style="background: linear-gradient(145deg, rgba(17,24,39,0.97) 0%, rgba(11,14,20,0.99) 100%)"
             (click)="$event.stopPropagation()">

          <div class="flex items-center justify-between mb-6">
            <h2 class="text-lg font-bold text-white">Launch Task</h2>
            <button (click)="showLaunchModal = false"
                    class="w-8 h-8 rounded-lg hover:bg-white/[0.06] flex items-center justify-center text-gray-400 transition">
              <svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2.5">
                <path stroke-linecap="round" d="M6 18L18 6M6 6l12 12"/>
              </svg>
            </button>
          </div>

          <div class="space-y-4">
            <div>
              <label class="block text-xs font-medium text-gray-400 mb-1.5">Task Type</label>
              <select [(ngModel)]="launchType" class="select-field">
                <option value="ablation">Ablation Study</option>
                <option value="pipeline">Pipeline (steps 7-10)</option>
                <option value="reports">Generate Reports</option>
                <option value="snapshot">Dashboard Snapshot</option>
              </select>
            </div>

            <div *ngIf="launchType === 'ablation'" class="space-y-4">
              <div>
                <label class="block text-xs font-medium text-gray-400 mb-1.5">Algorithm</label>
                <select [(ngModel)]="launchAlgorithm" class="select-field">
                  <option value="both">Both (PPO + SAC)</option>
                  <option value="ppo">PPO only</option>
                  <option value="sac">SAC only</option>
                </select>
              </div>
              <div class="grid grid-cols-2 gap-3">
                <div>
                  <label class="block text-xs font-medium text-gray-400 mb-1.5">Seeds</label>
                  <input type="number" [(ngModel)]="launchSeeds" min="1" max="20" class="input-field" />
                </div>
                <div>
                  <label class="block text-xs font-medium text-gray-400 mb-1.5">Timesteps</label>
                  <input type="number" [(ngModel)]="launchTimesteps" step="10000" min="5000" class="input-field" />
                </div>
              </div>
            </div>

            <div *ngIf="launchType === 'pipeline'">
              <label class="block text-xs font-medium text-gray-400 mb-1.5">Steps (e.g. 7,8,9,10 &mdash; leave blank for all)</label>
              <input type="text" [(ngModel)]="launchSteps" placeholder="7,8,9,10" class="input-field" />
            </div>
          </div>

          <div class="flex gap-3 mt-6">
            <button (click)="handleLaunch()" class="btn-primary flex-1">Launch</button>
            <button (click)="showLaunchModal = false" class="btn-secondary flex-1">Cancel</button>
          </div>
        </div>
      </div>
    </div>
  `,
})
export class TasksComponent implements OnInit, OnDestroy {
  Math = Math;

  tasks: TaskRun[] = [];
  activeTasks: TaskRun[] = [];
  historyTasks: TaskRun[] = [];
  expandedId: number | null = null;
  expandedTask: TaskRun | null = null;
  error: string | null = null;

  showLaunchModal = false;
  launchType = 'ablation';
  launchAlgorithm = 'both';
  launchSeeds = 5;
  launchTimesteps = 50000;
  launchSteps = '';

  private refreshSub?: Subscription;
  private detailSub?: Subscription;

  constructor(private api: ApiService) {}

  ngOnInit() {
    this.refresh();
    this.refreshSub = interval(5000).pipe(
      switchMap(() => this.api.getTasks(30))
    ).subscribe({
      next: d => this.processTasks(d.tasks),
      error: () => {},
    });
  }

  ngOnDestroy() {
    this.refreshSub?.unsubscribe();
    this.detailSub?.unsubscribe();
  }

  refresh() {
    this.api.getTasks(30).subscribe({
      next: d => this.processTasks(d.tasks),
      error: e => this.error = e.message,
    });
  }

  private processTasks(tasks: TaskRun[]) {
    this.tasks = tasks;
    this.activeTasks = tasks.filter(t => t.status === 'running' || t.status === 'pending');
    this.historyTasks = tasks.filter(t => t.status !== 'running' && t.status !== 'pending');
  }

  toggleExpanded(id: number) {
    this.detailSub?.unsubscribe();
    if (this.expandedId === id) {
      this.expandedId = null;
      this.expandedTask = null;
      return;
    }
    this.expandedId = id;
    this.loadDetail(id);
    this.detailSub = interval(3000).pipe(
      switchMap(() => this.expandedId === id ? this.api.getTaskDetail(id) : of(null))
    ).subscribe({
      next: t => { if (t) this.expandedTask = t; },
    });
  }

  private loadDetail(id: number) {
    this.api.getTaskDetail(id).subscribe({
      next: t => this.expandedTask = t,
      error: () => {},
    });
  }

  isHistoryExpanded(): boolean {
    return this.historyTasks.some(t => t.id === this.expandedId);
  }

  handleLaunch() {
    const config: Record<string, unknown> = {};
    if (this.launchType === 'ablation') {
      config['algorithm'] = this.launchAlgorithm;
      config['seeds'] = this.launchSeeds;
      config['timesteps'] = this.launchTimesteps;
    } else if (this.launchType === 'pipeline' && this.launchSteps) {
      config['steps'] = this.launchSteps;
    }

    const label = this.launchType === 'ablation'
      ? `Ablation ${this.launchAlgorithm.toUpperCase()} ${this.launchSeeds} seeds ${(this.launchTimesteps / 1000).toFixed(0)}k`
      : this.launchType === 'pipeline'
      ? `Pipeline steps ${this.launchSteps || 'all'}`
      : this.launchType.charAt(0).toUpperCase() + this.launchType.slice(1);

    this.api.launchTask(this.launchType, config, label).subscribe({
      next: () => { this.showLaunchModal = false; this.refresh(); },
      error: e => this.error = e.message,
    });
  }

  handleCancel(id: number) {
    this.api.cancelTask(id).subscribe({
      next: () => this.refresh(),
      error: e => this.error = e.message,
    });
  }

  fmtDuration(sec: number | null | undefined): string {
    if (sec == null) return '\u2014';
    if (sec < 60) return `${Math.round(sec)}s`;
    if (sec < 3600) return `${Math.floor(sec / 60)}m ${Math.round(sec % 60)}s`;
    const h = Math.floor(sec / 3600);
    const m = Math.floor((sec % 3600) / 60);
    return `${h}h ${m}m`;
  }

  fmtDate(s: string): string {
    try { return new Date(s).toLocaleString(); }
    catch { return s; }
  }

  statusBadge(s: string): string {
    switch (s) {
      case 'running': return 'badge-blue';
      case 'completed': return 'badge-green';
      case 'failed': return 'badge-red';
      case 'cancelled': return 'badge-yellow';
      default: return 'badge-gray';
    }
  }
}
