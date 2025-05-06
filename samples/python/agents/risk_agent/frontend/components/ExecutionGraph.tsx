"use client";

import React, { useEffect, useState } from "react";
import ReactFlow, { Background, Controls, Edge, Node } from "reactflow";
import "reactflow/dist/style.css";
import ActionNode from "./ActionNode";
import HypothesisNode from "./HypothesisNode";

// Props の型定義
interface ExecutionGraphProps {
  nodes: Node[];
  edges: Edge[];
}

// React Flow に渡すカスタムノード定義
const nodeTypes = {
  action: ActionNode,
  hypothesis: HypothesisNode,
};

// デフォルトエクスポートとして ExecutionGraph を定義
export default function ExecutionGraph({ nodes = [], edges = [] }: ExecutionGraphProps) {
  const [rfInstance, setRfInstance] = useState<any | null>(null);

  // ノード・エッジが更新されるたびに画面全体にフィット
  useEffect(() => {
    if (rfInstance && nodes && nodes.length > 0) {
      try { rfInstance.fitView({ padding: 0.3 }); } catch (e) { /* noop */ }
    }
  }, [nodes, edges, rfInstance]);

  // nodes が空配列の場合のフォールバック表示
  if (!nodes || nodes.length === 0) {
    return (
      <div style={{
        height: "100%",
        width: "100%",
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        color: '#aaa',
        fontSize: '14px',
        textAlign: 'center',
        border: '1px dashed #ccc',
        borderRadius: '4px',
        padding: '20px'
      }}>
        Waiting for execution data to generate graph...
      </div>
    );
  }

  return (
    <div style={{ height: "100%", width: "100%" }}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        minZoom={0.2}
        maxZoom={2}
        nodesDraggable={true} // ノードをドラッグ可能にする
        nodesConnectable={false} // ノード接続は不可
        elementsSelectable={true} // 要素を選択可能にする
        onInit={setRfInstance}
      >
        <Background />
        <Controls />
      </ReactFlow>
    </div>
  );
} 