import { Injectable } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';
import { Observable, throwError, timer } from 'rxjs';
import { catchError, timeout, retry } from 'rxjs/operators';

/* ---------- Types ---------- */

export interface TaskRun {
  id: number;
  task_type: string;
  task_label: string | null;
  status: string;
  progress_pct: number;
  current_step: number;
  total_steps: number;
  detail: string | null;
  log_tail: string | null;
  error_message: string | null;
  config_json: Record<string, unknown> | null;
  started_at: string | null;
  completed_at: string | null;
  created_at: string;
  elapsed_seconds: number | null;
  eta_seconds: number | null;
  pid: number | null;
}

export interface SignalFeedEntry {
  ts: string;
  underlying: string;
  news: { ts?: string; source?: string; text?: string; compound?: number; model?: string }[];
  pm_signals: {
    event_name: string;
    platform: string;
    token_id?: string;
    probability?: number;
    delta_1h?: number | null;
    affected_underlyings?: string[];
  }[];
  action: { delta_change?: number; vega_change?: number } | null;
  outcome: { pnl?: number } | null;
  iv: { atm_iv_7d?: number | null; atm_iv_14d?: number | null; atm_iv_30d?: number | null };
}

export interface OverviewData {
  kpis: {
    feature_bars: number;
    pm_prices: number;
    pm_events: number;
    sentiment_scored: number;
    latest_ts: string | null;
  };
  pipeline_status: { table: string; status: string }[];
  feature_coverage_7d: { bars: number; iv: number; pm: number; sentiment: number; equity: number } | null;
  last_pipeline_line: string | null;
  sentiment_summary: string | null;
}

export interface PerformanceData {
  equity: {
    ts: string[];
    buy_and_hold: number[];
    variant_d: number[];
  } | null;
  exposure: {
    delta: number[];
    vega: number[];
  } | null;
}

export interface AblationEntry {
  variant: string;
  algorithm: string;
  sharpe_mean: number;
  sharpe_std: number;
  max_drawdown_mean: number;
  hit_rate_mean: number;
}

/* ---------- Service ---------- */

@Injectable({ providedIn: 'root' })
export class ApiService {
  private readonly TIMEOUT = 20000;
  private readonly LONG_TIMEOUT = 60000;

  constructor(private http: HttpClient) {}

  /* -- Overview -- */
  getOverview(): Observable<OverviewData> {
    return this.http.get<OverviewData>('/api/overview').pipe(
      timeout(this.TIMEOUT),
      retry({ count: 1, delay: () => timer(2000) }),
    );
  }

  /* -- Signals -- */
  getSignalFeed(underlying: string, limit: number, positionChangesOnly: boolean): Observable<{ feed: SignalFeedEntry[] }> {
    const params = new HttpParams()
      .set('underlying', underlying)
      .set('limit', limit.toString())
      .set('position_changes_only', positionChangesOnly.toString());
    return this.http.get<{ feed: SignalFeedEntry[] }>('/api/signal-feed', { params }).pipe(
      timeout(this.LONG_TIMEOUT),
    );
  }

  /* -- Trade Impact (precomputed) -- */
  getTradeImpact(underlying: string, limit: number, showAll: boolean): Observable<{ bars: SignalFeedEntry[] }> {
    const params = new HttpParams()
      .set('underlying', underlying)
      .set('limit', limit.toString())
      .set('show_all', showAll.toString());
    return this.http.get<{ bars: SignalFeedEntry[] }>('/api/trade-impact', { params }).pipe(
      timeout(this.TIMEOUT),
    );
  }

  /* -- Performance -- */
  getPerformance(underlying: string): Observable<PerformanceData> {
    const params = new HttpParams().set('underlying', underlying);
    return this.http.get<PerformanceData>('/api/performance', { params }).pipe(
      timeout(this.TIMEOUT),
    );
  }

  getAblation(): Observable<{ aggregated: AblationEntry[] }> {
    return this.http.get<{ aggregated: AblationEntry[] }>('/api/ablation').pipe(
      timeout(this.TIMEOUT),
    );
  }

  /* -- Tasks -- */
  getTasks(limit = 30): Observable<{ tasks: TaskRun[] }> {
    const params = new HttpParams().set('limit', limit.toString());
    return this.http.get<{ tasks: TaskRun[] }>('/api/tasks', { params }).pipe(
      timeout(this.TIMEOUT),
    );
  }

  getTaskDetail(taskId: number): Observable<TaskRun> {
    return this.http.get<TaskRun>(`/api/tasks/${taskId}`).pipe(
      timeout(this.TIMEOUT),
    );
  }

  launchTask(taskType: string, config: Record<string, unknown>, label?: string): Observable<{ task_id: number; pid: number; status: string }> {
    return this.http.post<{ task_id: number; pid: number; status: string }>('/api/tasks', {
      task_type: taskType,
      config,
      label: label ?? null,
    }).pipe(timeout(this.TIMEOUT));
  }

  cancelTask(taskId: number): Observable<{ task_id: number; status: string }> {
    return this.http.post<{ task_id: number; status: string }>(`/api/tasks/${taskId}/cancel`, {}).pipe(
      timeout(this.TIMEOUT),
    );
  }
}
