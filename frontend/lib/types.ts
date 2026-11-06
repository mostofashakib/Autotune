/** Shapes the Autotune API returns, generated from its OpenAPI schema. Run `npm run gen:types` after the API changes. */
import type { components } from "./api-schema";

type Schemas = components["schemas"];

export type SystemStatus = Schemas["SystemStatus"];
export type Recipe = Schemas["Recipe"];
export type UploadedDataset = Schemas["UploadedDataset"];
export type TrainingSummary = Schemas["TrainingSummary"];
export type TrainingRun = Schemas["TrainingRun"];
export type ModelEndpoint = Schemas["ModelEndpoint"];
export type ModelScore = Schemas["ModelScore"];
export type TaskComparison = Schemas["TaskComparison"];
export type ComparisonReport = Schemas["ComparisonReport"];
export type BenchmarkSummary = Schemas["BenchmarkSummary"];
export type BenchmarkRun = Schemas["BenchmarkRun"];
export type ServingState = Schemas["ServingStatus"];
export type Gpu = Schemas["Gpu"];
export type Pod = Schemas["Pod"];
export type CredentialStatus = Schemas["CredentialStatus"];
export type CatalogModel = Schemas["CatalogModel"];
