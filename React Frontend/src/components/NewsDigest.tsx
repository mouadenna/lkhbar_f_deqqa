import React, { useState, useEffect } from 'react';
import { ChevronDown, ChevronUp, ExternalLink, Calendar, Hash, BookOpen, Building2, Filter, Search, RefreshCw, Newspaper, TrendingUp } from 'lucide-react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Skeleton } from "@/components/ui/skeleton";

interface TemporalSpan {
  start: string;
  end: string;
  duration_days: number;
}

interface Metrics {
  article_count: number;
  unique_sources: number;
  temporal_span: TemporalSpan;
  average_article_length: number;
}

interface SourceInfo {
  title: string;
  url: string;
  source: string;
  date: string;
}

interface Cluster {
  summary: string;
  keywords: string[];
  metrics: Metrics;
  sources_info: SourceInfo[];
}

interface Category {
  total_articles: number;
  clusters: Cluster[];
}

interface DigestData {
  [category: string]: Category;
}

const NewsDigest: React.FC = () => {
  const [data, setData] = useState<DigestData | null>(null);
  const [expandedCategory, setExpandedCategory] = useState<string | null>(null);
  const [expandedClusters, setExpandedClusters] = useState<Record<string, boolean>>({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedTimeRange, setSelectedTimeRange] = useState('all');
  const [filteredData, setFilteredData] = useState<DigestData | null>(null);
  const [viewMode, setViewMode] = useState<'compact' | 'expanded'>('compact');

  useEffect(() => {
    const loadData = async () => {
      try {
        const response = await fetch('/data.json');
        if (!response.ok) {
          throw new Error('Failed to load data');
        }
        const jsonData = await response.json();
        setData(jsonData);
        setFilteredData(jsonData);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'An error occurred');
      } finally {
        setLoading(false);
      }
    };

    loadData();
  }, []);

  useEffect(() => {
    if (!data) return;

    const filterData = () => {
      const filtered: DigestData = {};
      
      Object.entries(data).forEach(([categoryName, category]) => {
        const filteredClusters = category.clusters.filter(cluster => {
          const matchesSearch = searchTerm === '' || 
            cluster.summary.toLowerCase().includes(searchTerm.toLowerCase()) ||
            cluster.keywords.some(k => k.toLowerCase().includes(searchTerm.toLowerCase()));

          const matchesTimeRange = selectedTimeRange === 'all' || 
            isWithinTimeRange(cluster.metrics.temporal_span.end, selectedTimeRange);

          return matchesSearch && matchesTimeRange;
        });

        if (filteredClusters.length > 0) {
          filtered[categoryName] = {
            ...category,
            clusters: filteredClusters,
            total_articles: filteredClusters.reduce((sum, cluster) => 
              sum + cluster.metrics.article_count, 0
            )
          };
        }
      });

      setFilteredData(filtered);
    };

    filterData();
  }, [data, searchTerm, selectedTimeRange]);

  const isWithinTimeRange = (dateStr: string, range: string): boolean => {
    const date = new Date(dateStr);
    const now = new Date();
    const daysDiff = Math.floor((now.getTime() - date.getTime()) / (1000 * 60 * 60 * 24));

    switch (range) {
      case '7days': return daysDiff <= 7;
      case '30days': return daysDiff <= 30;
      case '90days': return daysDiff <= 90;
      default: return true;
    }
  };

  const formatDate = (dateStr: string): string => {
    return new Date(dateStr).toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric'
    });
  };

  const toggleCategory = (categoryName: string) => {
    setExpandedCategory(expandedCategory === categoryName ? null : categoryName);
  };

  const toggleCluster = (categoryName: string, index: number) => {
    const key = `${categoryName}-${index}`;
    setExpandedClusters(prev => ({
      ...prev,
      [key]: !prev[key]
    }));
  };

  const getRelevanceColor = (articleCount: number): string => {
    if (articleCount >= 10) return 'bg-red-500/10 text-red-700 dark:text-red-400';
    if (articleCount >= 5) return 'bg-orange-500/10 text-orange-700 dark:text-orange-400';
    return 'bg-blue-500/10 text-blue-700 dark:text-blue-400';
  };

  const renderCluster = (cluster: Cluster, categoryName: string, index: number) => {
    const isExpanded = expandedClusters[`${categoryName}-${index}`];
    const relevanceColor = getRelevanceColor(cluster.metrics.article_count);
    
    return (
      <Card key={index} className="mb-4 overflow-hidden border border-border/50 hover:border-border transition-colors">
        <CardHeader 
          className="cursor-pointer transition-colors hover:bg-accent/50" 
          onClick={() => toggleCluster(categoryName, index)}
        >
          <div className="flex justify-between items-start gap-4">
            <div className="flex-1">
              <div className="flex items-center gap-2 mb-2">
                <Badge variant="secondary" className={`${relevanceColor}`}>
                  {cluster.metrics.article_count} articles
                </Badge>
                <Badge variant="outline" className="text-xs">
                  {formatDate(cluster.metrics.temporal_span.start)} - {formatDate(cluster.metrics.temporal_span.end)}
                </Badge>
              </div>
              <CardTitle className="text-lg font-semibold leading-tight mb-2">
                {cluster.summary}
              </CardTitle>
              <div className="flex flex-wrap gap-2 mt-2">
                {cluster.keywords.map((keyword, idx) => (
                  <Badge key={idx} variant="secondary" className="text-xs bg-secondary/40">
                    {keyword}
                  </Badge>
                ))}
              </div>
            </div>
            <Button variant="ghost" size="icon" className="shrink-0">
              {isExpanded ? <ChevronUp className="w-5 h-5" /> : <ChevronDown className="w-5 h-5" />}
            </Button>
          </div>
        </CardHeader>
        
        {isExpanded && (
          <CardContent className="bg-card/50">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
              <div className="flex items-center gap-2 p-3 rounded-lg bg-secondary/20">
                <Hash className="w-4 h-4 text-muted-foreground" />
                <span className="text-sm font-medium">{cluster.metrics.article_count} articles</span>
              </div>
              <div className="flex items-center gap-2 p-3 rounded-lg bg-secondary/20">
                <Building2 className="w-4 h-4 text-muted-foreground" />
                <span className="text-sm font-medium">{cluster.metrics.unique_sources} sources</span>
              </div>
              <div className="flex items-center gap-2 p-3 rounded-lg bg-secondary/20">
                <Calendar className="w-4 h-4 text-muted-foreground" />
                <span className="text-sm font-medium">{cluster.metrics.temporal_span.duration_days} days span</span>
              </div>
              <div className="flex items-center gap-2 p-3 rounded-lg bg-secondary/20">
                <BookOpen className="w-4 h-4 text-muted-foreground" />
                <span className="text-sm font-medium">~{Math.round(cluster.metrics.average_article_length)} chars/article</span>
              </div>
            </div>
            
            <Separator className="my-4" />
            
            <div className="space-y-3">
              {cluster.sources_info.map((source, idx) => (
                <div key={idx} className="group">
                  <a 
                    href={source.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="block p-4 rounded-lg bg-secondary/10 hover:bg-secondary/20 transition-colors"
                  >
                    <div className="flex justify-between items-start gap-4">
                      <div className="flex-1 min-w-0">
                        <h4 className="font-medium text-primary group-hover:text-primary/80 transition-colors line-clamp-2">
                          {source.title}
                        </h4>
                        <div className="flex items-center gap-2 text-sm text-muted-foreground mt-2">
                          <Newspaper className="w-4 h-4" />
                          <span>{source.source}</span>
                          <span className="text-muted-foreground/50">•</span>
                          <Calendar className="w-4 h-4" />
                          <span>{formatDate(source.date)}</span>
                        </div>
                      </div>
                      <ExternalLink className="w-4 h-4 text-muted-foreground group-hover:text-primary transition-colors" />
                    </div>
                  </a>
                </div>
              ))}
            </div>
          </CardContent>
        )}
      </Card>
    );
  };

  if (error) {
    return (
      <Alert variant="destructive">
        <AlertDescription>{error}</AlertDescription>
      </Alert>
    );
  }

  return (
    <div className="max-w-5xl mx-auto p-6">
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-8">
        <h1 className="text-3xl font-bold">News Digest</h1>
        <div className="flex items-center gap-2">
          <Button 
            variant={viewMode === 'compact' ? 'default' : 'outline'}
            size="sm"
            onClick={() => setViewMode('compact')}
          >
            Compact
          </Button>
          <Button 
            variant={viewMode === 'expanded' ? 'default' : 'outline'}
            size="sm"
            onClick={() => setViewMode('expanded')}
          >
            Expanded
          </Button>
        </div>
      </div>
      
      <Card className="mb-8">
        <CardContent className="pt-6">
          <div className="flex flex-col md:flex-row gap-4">
            <div className="flex-1">
              <div className="relative">
                <Search className="absolute left-3 top-3 h-4 w-4 text-muted-foreground" />
                <Input
                  placeholder="Search articles..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="pl-10"
                />
              </div>
            </div>
            <Select value={selectedTimeRange} onValueChange={setSelectedTimeRange}>
              <SelectTrigger className="w-[180px]">
                <SelectValue placeholder="Time range" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All time</SelectItem>
                <SelectItem value="7days">Last 7 days</SelectItem>
                <SelectItem value="30days">Last 30 days</SelectItem>
                <SelectItem value="90days">Last 90 days</SelectItem>
              </SelectContent>
            </Select>
            <Button 
              variant="outline" 
              size="icon" 
              onClick={() => {
                setSearchTerm('');
                setSelectedTimeRange('all');
              }}
            >
              <RefreshCw className="h-4 w-4" />
            </Button>
          </div>
        </CardContent>
      </Card>

      {loading ? (
        <div className="space-y-6">
          {[1, 2, 3].map((i) => (
            <div key={i} className="space-y-4">
              <Skeleton className="h-12 w-full rounded-lg" />
              <Skeleton className="h-32 w-full rounded-lg" />
            </div>
          ))}
        </div>
      ) : filteredData && Object.keys(filteredData).length > 0 ? (
        Object.entries(filteredData).map(([categoryName, category]) => (
          <div key={categoryName} className="mb-8">
            <div 
              className="flex items-center justify-between cursor-pointer bg-primary/5 p-4 rounded-lg mb-4 hover:bg-primary/10 transition-colors"
              onClick={() => toggleCategory(categoryName)}
            >
              <div className="flex items-center gap-3">
                <div className="p-2 rounded-md bg-primary/10">
                  <TrendingUp className="w-5 h-5" />
                </div>
                <div>
                  <h2 className="text-xl font-semibold">{categoryName}</h2>
                  <p className="text-sm text-muted-foreground">
                    {category.total_articles} articles in {category.clusters.length} clusters
                  </p>
                </div>
              </div>
              {expandedCategory === categoryName ? (
                <ChevronUp className="w-6 h-6" />
              ) : (
                <ChevronDown className="w-6 h-6" />
              )}
            </div>
            
            {(expandedCategory === categoryName || viewMode === 'expanded') && (
              <div className="space-y-4">
                {category.clusters.map((cluster, index) => 
                  renderCluster(cluster, categoryName, index)
                )}
              </div>
            )}
          </div>
        ))
      ) : (
        <Alert>
          <AlertDescription>
            No articles found matching your search criteria.
          </AlertDescription>
        </Alert>
      )}
    </div>
  );
};

export default NewsDigest;