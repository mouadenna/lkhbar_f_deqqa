import { ThemeProvider } from "next-themes";
import NewsDigest from "./components/NewsDigest";

const App = () => {
  return (
    <ThemeProvider
      attribute="class"
      defaultTheme="system"
      enableSystem
      disableTransitionOnChange
    >
      <div className="min-h-screen bg-background text-foreground">
        <main className="relative py-6 lg:gap-10">
          <NewsDigest />
        </main>
      </div>
    </ThemeProvider>
  );
};

export default App;