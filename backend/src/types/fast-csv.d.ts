declare module 'fast-csv' {
  import { Readable, Writable, Transform } from 'stream';

  interface ParseOptions {
    headers?: boolean;
    delimiter?: string;
    quote?: string;
    escape?: string;
    skipLines?: number;
    maxRows?: number;
  }

  interface ParseStream extends Transform {
    pipe(destination: Writable): Writable;
    on(event: string, listener: Function): this;
    destroy(): void;
  }

  interface CSV {
    parse(options?: ParseOptions): ParseStream;
  }

  const csv: CSV;
  export = csv;
} 