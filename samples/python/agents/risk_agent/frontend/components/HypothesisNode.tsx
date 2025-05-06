import React from "react";
import { Handle, Position } from "reactflow";
import type { NodeProps } from "reactflow";

/*
 * シンプルな Hypothesis ノード。
 * label には改行が入るので <pre> を使用してそのまま表示。
 */
export default function HypothesisNode({ data }: NodeProps<Record<string, any>>) {
  return (
    <>
      <Handle type="target" position={Position.Left} />
      <pre style={{ margin: 0, whiteSpace: "pre-wrap", textAlign: "center", padding: 6 }}>
        {data?.label}
      </pre>
      <Handle type="source" position={Position.Right} />
    </>
  );
} 