import Providers from "./providers";
import type { ReactNode } from "react";

export const metadata = {
  title: "Risk Agent Dashboard",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="ja">
      <body>
        <Providers>{children}</Providers>
      </body>
    </html>
  );
} 