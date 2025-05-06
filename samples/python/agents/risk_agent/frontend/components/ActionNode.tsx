import React from "react";
import { Handle, Position } from "reactflow";
import type { NodeProps } from "reactflow";
import { Tooltip } from "@chakra-ui/react";

/*
 * シンプルな Action ノード。
 * 背景色などの装飾は ExecutionGraph で設定した style が
 * 自動的に適用されるため、ここではラベルとハンドルのみ描画する。
 */
export default function ActionNode({ data }: NodeProps<Record<string, any>>) {
  return (
    <>
      {/* 入力側ハンドル */}
      <Handle type="target" position={Position.Left} />
      {/* 本体 */}
      {data?.evaluationReason ? (
        <Tooltip label={data.evaluationReason} placement="top" hasArrow>
          <div style={{ whiteSpace: "pre-wrap", textAlign: "center", padding: 4 }}>
            {data?.label}
          </div>
        </Tooltip>
      ) : (
        <div style={{ whiteSpace: "pre-wrap", textAlign: "center", padding: 4 }}>
          {data?.label}
        </div>
      )}
      {/* 出力側ハンドル */}
      <Handle type="source" position={Position.Right} />
    </>
  );
} 