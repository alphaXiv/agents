import { Model } from "../model.ts";
import { SidAdapter, type SidAdapterOptions, type SidModels } from "./adapter.ts";

export type SidModelOptions<TModel extends SidModels> = Omit<SidAdapterOptions<TModel>, "client">;

export class SidModel<TModel extends SidModels = SidModels> extends Model<TModel> {
  adapter: SidAdapter<TModel>;

  constructor(options: SidModelOptions<TModel>) {
    super(options);
    this.adapter = new SidAdapter(options);
  }
}
