#!/usr/bin/env python
"""复用视频基线 CSV，比较连续 N 个采样帧有烟的规则，无须重复推理。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def save_csv(path, rows):
    if rows:
        with path.open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def load_annotations(path, videos):
    intervals = defaultdict(list)
    if path is None:
        return intervals
    for row in read_csv(path):
        video = row['video_id']
        start, end = int(row['start_frame']), int(row['end_frame'])
        state = row['state']
        onset = int(row['smoke_onset_frame']) if row.get('smoke_onset_frame') else None
        if video not in videos or start < 0 or end < start or state not in {'smoke', 'clear', 'unknown'}:
            raise ValueError(f'标注视频、范围或状态无效: {row}')
        if onset is not None and (state != 'smoke' or not start <= onset <= end):
            raise ValueError('起烟帧只能填写在 smoke 片段内')
        intervals[video].append(dict(start=start, end=end, state=state, onset=onset,
                                     interference=row.get('interference', '')))
    for entries in intervals.values():
        entries.sort(key=lambda item: item['start'])
        if any(a['end'] >= b['start'] for a, b in zip(entries, entries[1:])):
            raise ValueError('标注区间不能重叠；首尾帧均包含在区间内')
    return intervals


def evaluate(detections, out, consecutive=(1, 3, 5), annotations=None):
    consecutive = sorted(set(consecutive))
    if not consecutive or any(n < 1 for n in consecutive):
        raise ValueError('连续帧数必须 >=1')
    groups = defaultdict(list)
    for row in read_csv(detections):
        row = dict(row, frame_index=int(row['frame_index']), timestamp_s=float(row['timestamp_s']),
                   fps=float(row['fps']), stride=int(row['stride']), smoke_boxes=int(row['smoke_boxes']))
        row['decision_timestamp_s'] = float(row.get('decision_timestamp_s', row['timestamp_s']))
        if (row['frame_index'] < 0 or row['stride'] < 1 or row['smoke_boxes'] < 0
                or not math.isfinite(row['fps']) or row['fps'] <= 0
                or not math.isfinite(row['timestamp_s']) or row['timestamp_s'] < 0
                or not math.isfinite(row['decision_timestamp_s']) or row['decision_timestamp_s'] < row['timestamp_s']):
            raise ValueError('检测 CSV 含无效帧号、时间戳、FPS、stride 或框数')
        groups[(row['model'], row['video_id'])].append(row)
    if not groups:
        raise ValueError('检测 CSV 为空')
    labels = load_annotations(annotations, {video for _, video in groups})
    summaries, events, frames, segments = [], [], [], []
    # 新目录避免重跑时覆盖旧实验；也避免失败后混用旧 CSV。
    out.mkdir(parents=True, exist_ok=False)
    for (model, video), rows in groups.items():
        rows.sort(key=lambda row: row['frame_index'])
        fps, stride = rows[0]['fps'], rows[0]['stride']
        if any(r['fps'] != fps or r['stride'] != stride for r in rows):
            raise ValueError('同一视频的 FPS / stride 必须一致')
        if len({r['frame_index'] for r in rows}) != len(rows):
            raise ValueError('同一模型/视频出现重复帧号')
        if any(b['timestamp_s'] <= a['timestamp_s'] for a, b in zip(rows, rows[1:])):
            raise ValueError('时间戳必须递增')
        labeled = []
        for row in rows:
            label = next((entry for entry in labels[video] if entry['start'] <= row['frame_index'] <= entry['end']), None)
            labeled.append(label)
        for n in consecutive:
            run, previous, active, first = 0, None, False, None
            active_frames, starts = [], []
            for row, label in zip(rows, labeled):
                index = row['frame_index']
                if previous is None or index - previous != stride:
                    run, active = 0, False
                run = run + 1 if row['smoke_boxes'] else 0
                alarm = run >= n
                triggered = alarm and not active
                if triggered:
                    starts.append(index)
                    if first is None:
                        first = row['decision_timestamp_s']
                    events.append(dict(model=model, video_id=video, consecutive=n, frame_index=index,
                                       timestamp_s=row['decision_timestamp_s'], target_timestamp_s=row['timestamp_s'],
                                       run_start_frame=index - (run - 1) * stride,
                                       annotation_state=label['state'] if label else 'unknown'))
                active_frames.append(alarm)
                frames.append(dict(model=model, video_id=video, consecutive=n, frame_index=index,
                                   timestamp_s=row['timestamp_s'], smoke_hit=int(row['smoke_boxes'] > 0),
                                   decision_timestamp_s=row['decision_timestamp_s'],
                                   streak=run, alarm_active=int(alarm), alarm_start=int(triggered),
                                   annotation_state=label['state'] if label else 'unknown',
                                   interference=label['interference'] if label else ''))
                active, previous = alarm, index
            clear = [i for i, label in enumerate(labeled) if label and label['state'] == 'clear']
            smoke = [i for i, label in enumerate(labeled) if label and label['state'] == 'smoke']
            summaries.append(dict(model=model, video_id=video, consecutive=n, stride=stride, fps=fps,
                                  frames=len(rows), alarm_events=len(starts), alarm_active_frames=sum(active_frames),
                                  first_trigger_s=first if first is not None else '',
                                  max_lookahead_wait_s=max(r['decision_timestamp_s'] - r['timestamp_s'] for r in rows),
                                  labeled_smoke_frames=len(smoke), labeled_clear_frames=len(clear),
                                  smoke_frame_alarm_rate=sum(active_frames[i] for i in smoke) / len(smoke) if smoke else '',
                                  clear_frame_alarm_rate=sum(active_frames[i] for i in clear) / len(clear) if clear else '',
                                  theoretical_min_extra_wait_s=(n - 1) * stride / fps))
            for label in labels[video]:
                if label['state'] != 'smoke':
                    continue
                indices = [i for i, row in enumerate(rows) if label['start'] <= row['frame_index'] <= label['end']]
                # 若起烟之前已持续报警，不视作该起烟事件的有效检测；需重新触发。
                onset = label['onset']
                eligible = [i for i in indices if active_frames[i]]
                effective = [row for row in rows if row['frame_index'] in starts and onset is not None
                             and onset + (n - 1) * stride <= row['frame_index'] <= label['end']]
                first_effective = effective[0]['decision_timestamp_s'] if effective else None
                segments.append(dict(model=model, video_id=video, consecutive=n, start_frame=label['start'],
                                     end_frame=label['end'], sampled_frames=len(indices),
                                     segment_alarm_hit=int(bool(eligible)) if indices else '',
                                     smoke_onset_frame=onset if onset is not None else '',
                                     first_effective_trigger_s=first_effective if first_effective is not None else '',
                                     delay_s=first_effective - onset / fps if first_effective is not None else '',
                                     timing_status='measured' if first_effective is not None else
                                     'onset_unknown' if onset is None else 'no_new_trigger'))
    save_csv(out / 'summary.csv', summaries)
    save_csv(out / 'frames.csv', frames)
    # 即使零次触发也生成带表头文件，便于明确区分“无触发”与“缺少结果”。
    event_fields = ['model', 'video_id', 'consecutive', 'frame_index', 'timestamp_s', 'target_timestamp_s', 'run_start_frame', 'annotation_state']
    with (out / 'events.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=event_fields)
        writer.writeheader()
        writer.writerows(events)
    save_csv(out / 'smoke_segments.csv', segments)
    (out / 'settings.json').write_text(json.dumps(dict(
        consecutive=consecutive, annotations_used=annotations is not None,
        detections_sha256=hashlib.sha256(detections.read_bytes()).hexdigest(),
        annotations_sha256=hashlib.sha256(annotations.read_bytes()).hexdigest() if annotations else None,
        rule='scene-level smoke only; N consecutive sampled frames; reset on miss/gap/video boundary; no latch or tracking',
        limitations='unknown labels excluded; first_trigger_s is clip time, not early-warning delay; nominal FPS assumes CFR; metrics are sampled-frame metrics',
    ), ensure_ascii=False, indent=2), encoding='utf-8')
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--detections', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True, help='必须是尚不存在的新目录')
    parser.add_argument('--consecutive', nargs='+', type=int, default=[1, 3, 5])
    parser.add_argument('--annotations', type=Path)
    args = parser.parse_args()
    evaluate(args.detections, args.out, args.consecutive, args.annotations)
    print(f'连续帧对照已输出: {args.out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
