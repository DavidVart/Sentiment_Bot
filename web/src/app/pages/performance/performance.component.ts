import { Component, OnInit, ViewChild, ElementRef, AfterViewInit, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ApiService, PerformanceData, AblationEntry } from '../../services/api.service';
import { Chart, registerables } from 'chart.js';

Chart.register(...registerables);

const VARIANT_LABELS: Record<string, string> = { A: 'Base', B: '+Sentiment', C: '+PM', D: 'Full' };

@Component({
  selector: 'app-performance',
  standalone: true,
  imports: [CommonModule, FormsModule],
  template: `
    <div class="space-y-6 animate-fade-in">
      <div class="flex flex-col sm:flex-row sm:items-center gap-4">
        <h1 class="page-title">Performance</h1>
        <input [(ngModel)]="underlying" (keyup.enter)="load()" placeholder="SPY"
               class="input-field w-24 text-sm !py-2" />
      </div>

      <!-- Loading -->
      <div *ngIf="loading" class="flex items-center gap-3 text-gray-400">
        <svg class="animate-spin w-5 h-5" fill="none" viewBox="0 0 24 24">
          <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
          <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"></path>
        </svg>
        Loading performance data...
      </div>

      <div *ngIf="error" class="glass-card border-red-500/20 text-red-400 text-sm">{{ error }}</div>

      <div *ngIf="!loading && !error">
        <!-- Equity Curve -->
        <div *ngIf="hasEquity" class="glass-card mb-6">
          <h3 class="section-title mb-4">Equity Curve</h3>
          <div class="relative" style="height: 350px">
            <canvas #equityCanvas></canvas>
          </div>
        </div>

        <!-- Sharpe Chart -->
        <div *ngIf="ppoAgg.length > 0" class="glass-card mb-6">
          <h3 class="section-title mb-4">Sharpe Ratio (Mean &plusmn; Std)</h3>
          <div class="relative" style="height: 300px">
            <canvas #sharpeCanvas></canvas>
          </div>
        </div>

        <!-- Ablation Table -->
        <div *ngIf="ppoAgg.length > 0" class="glass-card mb-6 overflow-x-auto">
          <h3 class="section-title mb-4">Ablation Results (PPO)</h3>
          <table class="w-full text-sm">
            <thead>
              <tr class="text-left text-gray-500 border-b border-white/[0.06]">
                <th class="pb-3 pr-6 font-medium">Variant</th>
                <th class="pb-3 pr-6 font-medium">Sharpe</th>
                <th class="pb-3 pr-6 font-medium">Max DD</th>
                <th class="pb-3 font-medium">Hit Rate %</th>
              </tr>
            </thead>
            <tbody>
              <tr *ngFor="let a of ppoAgg" class="border-b border-white/[0.03] hover:bg-white/[0.02] transition">
                <td class="py-3 pr-6 text-white font-medium">{{ variantLabel(a.variant) }}</td>
                <td class="py-3 pr-6 text-gray-300 tabular-nums">{{ a.sharpe_mean?.toFixed(2) }} &plusmn; {{ a.sharpe_std?.toFixed(2) }}</td>
                <td class="py-3 pr-6 text-gray-300 tabular-nums">{{ a.max_drawdown_mean?.toFixed(4) }}</td>
                <td class="py-3 text-gray-300 tabular-nums">{{ a.hit_rate_mean?.toFixed(2) }}</td>
              </tr>
            </tbody>
          </table>
        </div>

        <!-- Exposure -->
        <div *ngIf="hasExposure" class="glass-card mb-6">
          <h3 class="section-title mb-4">Exposure (Variant D)</h3>
          <div class="relative" style="height: 280px">
            <canvas #exposureCanvas></canvas>
          </div>
        </div>

        <!-- No data -->
        <div *ngIf="!hasEquity && ppoAgg.length === 0 && !hasExposure" class="glass-card text-center py-12">
          <p class="text-gray-500">No precomputed performance data.</p>
          <p class="text-sm text-gray-600 mt-1">Run the pipeline and snapshot script.</p>
        </div>
      </div>
    </div>
  `,
})
export class PerformanceComponent implements OnInit, OnDestroy {
  @ViewChild('equityCanvas') equityCanvas!: ElementRef<HTMLCanvasElement>;
  @ViewChild('sharpeCanvas') sharpeCanvas!: ElementRef<HTMLCanvasElement>;
  @ViewChild('exposureCanvas') exposureCanvas!: ElementRef<HTMLCanvasElement>;

  underlying = 'SPY';
  loading = true;
  error: string | null = null;
  perf: PerformanceData | null = null;
  ppoAgg: AblationEntry[] = [];
  hasEquity = false;
  hasExposure = false;

  private charts: Chart[] = [];

  constructor(private api: ApiService) {}

  ngOnInit() { this.load(); }

  ngOnDestroy() { this.charts.forEach(c => c.destroy()); }

  load() {
    this.loading = true;
    this.error = null;
    this.charts.forEach(c => c.destroy());
    this.charts = [];

    let loaded = 0;
    const done = () => { if (++loaded >= 2) { this.loading = false; setTimeout(() => this.renderCharts(), 50); } };

    this.api.getPerformance(this.underlying).subscribe({
      next: d => {
        this.perf = d;
        this.hasEquity = !!(d.equity?.ts?.length);
        this.hasExposure = !!(d.exposure?.delta?.length);
        done();
      },
      error: e => { this.error = e.message; done(); },
    });

    this.api.getAblation().subscribe({
      next: d => {
        this.ppoAgg = (d.aggregated || []).filter(a => a.algorithm === 'ppo');
        done();
      },
      error: () => done(),
    });
  }

  variantLabel(v: string): string { return VARIANT_LABELS[v] ?? v; }

  private renderCharts() {
    if (this.hasEquity && this.equityCanvas) {
      const eq = this.perf!.equity!;
      const labels = eq.ts.map(t => t.slice(5, 10));
      this.charts.push(new Chart(this.equityCanvas.nativeElement, {
        type: 'line',
        data: {
          labels,
          datasets: [
            { label: 'Buy & Hold', data: eq.buy_and_hold, borderColor: '#3b82f6', backgroundColor: 'rgba(59,130,246,0.08)', fill: true, pointRadius: 0, borderWidth: 2, tension: 0.3 },
            { label: 'Variant D (Full)', data: eq.variant_d, borderColor: '#10b981', backgroundColor: 'rgba(16,185,129,0.08)', fill: true, pointRadius: 0, borderWidth: 2, tension: 0.3 },
          ],
        },
        options: this.lineOpts(),
      }));
    }

    if (this.ppoAgg.length > 0 && this.sharpeCanvas) {
      this.charts.push(new Chart(this.sharpeCanvas.nativeElement, {
        type: 'bar',
        data: {
          labels: this.ppoAgg.map(a => this.variantLabel(a.variant)),
          datasets: [{
            label: 'Sharpe',
            data: this.ppoAgg.map(a => a.sharpe_mean),
            backgroundColor: ['rgba(59,130,246,0.6)', 'rgba(16,185,129,0.6)', 'rgba(139,92,246,0.6)', 'rgba(245,158,11,0.6)'],
            borderColor: ['#3b82f6', '#10b981', '#8b5cf6', '#f59e0b'],
            borderWidth: 1,
            borderRadius: 6,
          }],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { display: false } },
          scales: {
            x: { grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { color: '#9ca3af' } },
            y: { grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { color: '#9ca3af' } },
          },
        },
      }));
    }

    if (this.hasExposure && this.exposureCanvas) {
      const exp = this.perf!.exposure!;
      this.charts.push(new Chart(this.exposureCanvas.nativeElement, {
        type: 'line',
        data: {
          labels: exp.delta.map((_, i) => i.toString()),
          datasets: [
            { label: 'Net Delta', data: exp.delta, borderColor: '#3b82f6', pointRadius: 0, borderWidth: 1.5, tension: 0.2 },
            { label: 'Net Vega', data: exp.vega, borderColor: '#f59e0b', pointRadius: 0, borderWidth: 1.5, tension: 0.2 },
          ],
        },
        options: this.lineOpts(),
      }));
    }
  }

  private lineOpts(): any {
    return {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { labels: { color: '#9ca3af', usePointStyle: true, pointStyle: 'circle', padding: 20 } },
        tooltip: { backgroundColor: '#1f2937', titleColor: '#f9fafb', bodyColor: '#d1d5db', borderColor: 'rgba(255,255,255,0.1)', borderWidth: 1, padding: 12, cornerRadius: 8 },
      },
      scales: {
        x: { grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { color: '#9ca3af', maxTicksLimit: 12, font: { size: 10 } } },
        y: { grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { color: '#9ca3af' } },
      },
    };
  }
}
