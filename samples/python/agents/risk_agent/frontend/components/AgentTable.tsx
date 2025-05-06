"use client";
import { Table, Thead, Tr, Th, Tbody, Td, Badge } from "@chakra-ui/react";

interface Agent {
  agent_name: string;
  skill_id?: string | null;
  skill_name?: string | null;
  skill_description?: string | null;
}

export default function AgentTable({ agents = [] }: { agents: Agent[] }) {
  return (
    <Table size="sm" variant="striped">
      <Thead>
        <Tr>
          <Th>Name</Th>
          <Th>Skill</Th>
        </Tr>
      </Thead>
      <Tbody>
        {agents.map((a, index) => (
          <Tr key={a.skill_id ?? `${a.agent_name}-${index}`}>
            <Td>{a.agent_name}</Td>
            <Td>
              {a.skill_name ? (
                <Badge colorScheme="purple" variant="subtle">
                  {a.skill_name}
                </Badge>
              ) : (
                "-"
              )}
            </Td>
          </Tr>
        ))}
      </Tbody>
    </Table>
  );
} 